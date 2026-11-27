//! pool-iroh: the iroh-blobs sidecar of a pool agent (#45 sections 5 and 15, #89).
//!
//! The agent stays standard-library Python; this binary (or its container) does the peer-to-peer
//! part: QUIC between machines with hole punching and relays (iroh), and verified streaming of
//! blobs (iroh-blobs, BLAKE3). Authorization is unchanged: the pool server signs a ticket per
//! (machine, chunk), and only the holder's own agent checks it.
//!
//! Two sides, one endpoint:
//!
//!   control (agent -> its own sidecar), TCP on loopback, one JSON line per connection:
//!     {"op":"status"}                                   -> {"ok":true,"id":..,"addr":<EndpointAddr>}
//!     {"op":"fetch","peer":<EndpointAddr>,"sha":..,"ticket":..}
//!                                                       -> {"ok":true,"size":N} then N raw bytes
//!                                                          or {"ok":false,"error":..}
//!
//!   peers (sidecar -> sidecar):
//!     ALPN pool/chunk-auth/1: the fetcher sends {"sha","ticket"}. The holder asks its agent's peer
//!       server on loopback (GET /peer/chunks/<sha> with the ticket: the agent checks the ticket and
//!       returns the bytes), imports them, grants (fetcher endpoint, blake3) for a short while and
//!       answers {"ok":true,"hash":<blake3>,"size":N}.
//!     ALPN iroh-blobs: the fetcher gets that blob. The provider's event hook lets a get through only
//!       for a granted (endpoint, hash); get-many, push and observe are disabled.
//!
//! The fetcher's agent still checks the sha256 of what it receives, so neither side trusts the other.

use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

use anyhow::{anyhow, bail, Context, Result};
use iroh::{
    endpoint::{presets, Connection},
    protocol::{AcceptError, ProtocolHandler, Router},
    Endpoint, EndpointAddr, EndpointId, RelayMode,
};
use iroh_blobs::{
    api::{Store, TempTag},
    provider::events::{AbortReason, ConnectMode, EventMask, EventSender, ProviderMessage, RequestMode},
    store::{
        mem::{MemStore, Options},
        GcConfig,
    },
    BlobsProtocol, Hash, HashAndFormat,
};
use serde::{Deserialize, Serialize};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    net::{TcpListener, TcpStream},
};

const AUTH_ALPN: &[u8] = b"pool/chunk-auth/1";
const GRANT: Duration = Duration::from_secs(120);
const KEEP: Duration = Duration::from_secs(60);
const HELD_MAX: usize = 16; // chunks kept imported for other fetchers: at most 1 GiB of memory
const CHUNK_MAX: usize = 64 << 20;
const REQUEST_MAX: usize = 1 << 16;

type Grants = Arc<Mutex<HashMap<(EndpointId, Hash), Instant>>>;

struct Args {
    control: SocketAddr,
    agent: String,
    bind: Option<SocketAddr>,
    lan: bool,
    parent: Option<u32>,
}

fn args() -> Result<Args> {
    let mut out = Args { control: "127.0.0.1:8767".parse()?, agent: "127.0.0.1:8766".into(), bind: None, lan: false, parent: None };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || it.next().ok_or_else(|| anyhow!("{flag} needs a value"));
        match flag.as_str() {
            "--control" => out.control = value()?.parse()?,
            "--agent" => out.agent = value()?.trim_start_matches("http://").trim_end_matches('/').to_string(),
            "--bind" => out.bind = Some(value()?.parse()?),
            "--lan" => out.lan = true,
            "--parent" => out.parent = Some(value()?.parse()?),
            "-h" | "--help" => {
                println!("pool-iroh [--control 127.0.0.1:8767] [--agent 127.0.0.1:8766] [--bind 0.0.0.0:PORT] [--lan] [--parent PID]\n\
                          --lan: no relays and no address lookup (direct addresses only)\n\
                          --parent: exit when this process (the agent) is gone (Linux)");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(out)
}

#[derive(Deserialize)]
struct Control {
    op: String,
    peer: Option<EndpointAddr>,
    sha: Option<String>,
    ticket: Option<String>,
}

#[derive(Serialize, Deserialize)]
struct AuthRequest {
    sha: String,
    ticket: String,
}

#[derive(Serialize, Deserialize)]
struct AuthReply {
    ok: bool,
    #[serde(default)]
    hash: Option<String>,
    #[serde(default)]
    size: Option<u64>,
    #[serde(default)]
    error: Option<String>,
}

/// The provider side of the chunk-auth protocol.
#[derive(Clone, Debug)]
struct Auth {
    store: Store,
    agent: String,
    grants: Grants,
    /// sha256 -> (its blob, kept until): several fetchers of one chunk import it once
    held: Arc<Mutex<HashMap<String, (Arc<TempTag>, u64, Instant)>>>,
}

impl Auth {
    async fn grant(&self, remote: EndpointId, request: AuthRequest) -> Result<(Hash, u64)> {
        if request.sha.len() != 64 || !request.sha.bytes().all(|b| b.is_ascii_hexdigit()) {
            bail!("bad chunk id");
        }
        let now = Instant::now();
        let cached = {
            let mut held = self.held.lock().unwrap();
            held.retain(|_, (_, _, until)| *until > now);
            held.get(&request.sha).map(|(tag, size, _)| (tag.hash(), *size))
        };
        let (hash, size) = match cached {
            // Already imported: the agent still checks this fetcher's ticket.
            Some(found) => {
                agent_chunk(&self.agent, &request.sha, &request.ticket, true).await?;
                found
            }
            None => {
                let data = agent_chunk(&self.agent, &request.sha, &request.ticket, false).await?;
                let size = data.len() as u64;
                let tag = self.store.add_bytes(data).temp_tag().await?;
                let hash = tag.hash();
                let mut held = self.held.lock().unwrap();
                held.insert(request.sha.clone(), (Arc::new(tag), size, now + KEEP));
                while held.len() > HELD_MAX {
                    // memory stays bounded: the oldest import goes (should a transfer of it still be
                    // under way when GC runs, it fails and the fetcher's agent falls back to HTTP)
                    let oldest = held.iter().min_by_key(|(_, (_, _, until))| *until).map(|(sha, _)| sha.clone());
                    match oldest {
                        Some(sha) => held.remove(&sha),
                        None => break,
                    };
                }
                (hash, size)
            }
        };
        let mut grants = self.grants.lock().unwrap();
        grants.retain(|_, until| *until > now);
        grants.insert((remote, hash), now + GRANT);
        Ok((hash, size))
    }
}

impl ProtocolHandler for Auth {
    async fn accept(&self, connection: Connection) -> std::result::Result<(), AcceptError> {
        let remote = connection.remote_id();
        let (mut send, mut recv) = connection.accept_bi().await?;
        let raw = recv.read_to_end(REQUEST_MAX).await.map_err(AcceptError::from_err)?;
        let reply = match serde_json::from_slice::<AuthRequest>(&raw) {
            Ok(request) => match self.grant(remote, request).await {
                Ok((hash, size)) => AuthReply { ok: true, hash: Some(hash.to_hex()), size: Some(size), error: None },
                Err(err) => AuthReply { ok: false, hash: None, size: None, error: Some(err.to_string()) },
            },
            Err(err) => AuthReply { ok: false, hash: None, size: None, error: Some(format!("bad request: {err}")) },
        };
        send.write_all(&serde_json::to_vec(&reply).unwrap()).await.map_err(AcceptError::from_err)?;
        send.finish()?;
        connection.closed().await;
        Ok(())
    }
}

/// Ask the agent's peer server for a chunk (or, with `check`, only whether the ticket is good).
async fn agent_chunk(agent: &str, sha: &str, ticket: &str, check: bool) -> Result<Vec<u8>> {
    let mut stream = TcpStream::connect(agent).await.context("agent peer server unreachable")?;
    let method = if check { "HEAD" } else { "GET" };
    let head = format!("{method} /peer/chunks/{sha} HTTP/1.1\r\nHost: {agent}\r\nX-Pool-Ticket: {ticket}\r\nConnection: close\r\n\r\n");
    stream.write_all(head.as_bytes()).await?;
    let mut reader = BufReader::new(stream);
    let mut status = String::new();
    reader.read_line(&mut status).await?;
    let code: u16 = status.split_whitespace().nth(1).and_then(|c| c.parse().ok()).unwrap_or(0);
    let mut length = 0usize;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line).await? == 0 || line == "\r\n" {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(0);
            }
        }
    }
    if code != 200 {
        bail!("the holder's agent refused (HTTP {code})");
    }
    if check {
        return Ok(Vec::new());
    }
    if length > CHUNK_MAX {
        bail!("chunk larger than 64 MiB");
    }
    let mut body = vec![0u8; length];
    reader.read_exact(&mut body).await?;
    Ok(body)
}

/// Lets an iroh-blobs get through only for a hash granted to the connection's endpoint.
fn gate(grants: Grants) -> EventSender {
    let mask = EventMask {
        connected: ConnectMode::Intercept,
        get: RequestMode::Intercept,
        get_many: RequestMode::Disabled,
        ..EventMask::DEFAULT
    };
    let (tx, mut rx) = EventSender::channel(64, mask);
    tokio::spawn(async move {
        let mut connections: HashMap<u64, EndpointId> = HashMap::new();
        while let Some(msg) = rx.recv().await {
            match msg {
                ProviderMessage::ClientConnected(msg) => {
                    let res = match msg.endpoint_id {
                        Some(id) => {
                            connections.insert(msg.connection_id, id);
                            Ok(())
                        }
                        None => Err(AbortReason::Permission),
                    };
                    msg.tx.send(res).await.ok();
                }
                ProviderMessage::ConnectionClosed(msg) => {
                    connections.remove(&msg.connection_id);
                }
                ProviderMessage::GetRequestReceived(msg) => {
                    let allowed = msg.request.ranges.is_blob()
                        && connections.get(&msg.connection_id).is_some_and(|id| {
                            grants.lock().unwrap().get(&(*id, msg.request.hash)).is_some_and(|until| *until > Instant::now())
                        });
                    msg.tx.send(if allowed { Ok(()) } else { Err(AbortReason::Permission) }).await.ok();
                }
                _ => {}
            }
        }
    });
    tx
}

/// The fetcher side: authorize with the holder, then get the blob.
async fn fetch(endpoint: &Endpoint, store: &Store, peer: EndpointAddr, sha: String, ticket: String) -> Result<bytes::Bytes> {
    let auth = endpoint.connect(peer.clone(), AUTH_ALPN).await.context("connecting to the peer")?;
    let (mut send, mut recv) = auth.open_bi().await?;
    send.write_all(&serde_json::to_vec(&AuthRequest { sha, ticket })?).await?;
    send.finish()?;
    let reply: AuthReply = serde_json::from_slice(&recv.read_to_end(REQUEST_MAX).await?)?;
    auth.close(0u32.into(), b"done");
    if !reply.ok {
        bail!("peer refused: {}", reply.error.unwrap_or_default());
    }
    let hash: Hash = reply.hash.ok_or_else(|| anyhow!("no hash in the reply"))?.parse()?;
    let _keep = store.tags().temp_tag(HashAndFormat::raw(hash)).await?;
    let blobs = endpoint.connect(peer, iroh_blobs::ALPN).await.context("connecting for the blob")?;
    store.remote().fetch(blobs.clone(), hash).await?;
    blobs.close(0u32.into(), b"done");
    let data = store.get_bytes(hash).await?;
    if Some(data.len() as u64) != reply.size {
        bail!("size does not match");
    }
    Ok(data)
}

async fn control(stream: TcpStream, endpoint: Endpoint, store: Store) -> Result<()> {
    let (read, mut write) = stream.into_split();
    let mut line = String::new();
    BufReader::new(read).take(REQUEST_MAX as u64).read_line(&mut line).await?;
    let request: Control = serde_json::from_str(&line)?;
    match request.op.as_str() {
        "status" => {
            let reply = serde_json::json!({"ok": true, "id": endpoint.id().to_string(), "addr": endpoint.addr()});
            write.write_all(format!("{reply}\n").as_bytes()).await?;
        }
        "fetch" => {
            let got = match (request.peer, request.sha, request.ticket) {
                (Some(peer), Some(sha), Some(ticket)) => fetch(&endpoint, &store, peer, sha, ticket).await,
                _ => Err(anyhow!("fetch needs peer, sha and ticket")),
            };
            match got {
                Ok(data) => {
                    write.write_all(format!("{}\n", serde_json::json!({"ok": true, "size": data.len()})).as_bytes()).await?;
                    write.write_all(&data).await?;
                }
                Err(err) => {
                    write.write_all(format!("{}\n", serde_json::json!({"ok": false, "error": format!("{err:#}")})).as_bytes()).await?;
                }
            }
        }
        other => {
            write.write_all(format!("{}\n", serde_json::json!({"ok": false, "error": format!("unknown op {other}")})).as_bytes()).await?;
        }
    }
    write.shutdown().await.ok();
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = args()?;
    let store = MemStore::new_with_opts(Options {
        gc_config: Some(GcConfig { interval: Duration::from_secs(15), add_protected: None }),
    });
    let mut builder = if args.lan {
        Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled)
    } else {
        Endpoint::builder(presets::N0)
    };
    if let Some(bind) = args.bind {
        builder = builder.bind_addr(bind)?;
    }
    let endpoint = builder.bind().await?;
    if !args.lan {
        let _ = tokio::time::timeout(Duration::from_secs(10), endpoint.online()).await;
    }
    let grants: Grants = Default::default();
    let auth = Auth { store: (*store).clone(), agent: args.agent.clone(), grants: grants.clone(), held: Default::default() };
    let blobs = BlobsProtocol::new(&store, Some(gate(grants)));
    let router = Router::builder(endpoint).accept(AUTH_ALPN, auth).accept(iroh_blobs::ALPN, blobs).spawn();
    let endpoint = router.endpoint().clone();
    let listener = TcpListener::bind(args.control).await?;
    eprintln!("pool-iroh {} control on {}, agent at {}", endpoint.id(), args.control, args.agent);
    let parent = args.parent;
    let orphaned = async move {
        match parent {
            Some(pid) => loop {
                tokio::time::sleep(Duration::from_secs(2)).await;
                if !std::path::Path::new(&format!("/proc/{pid}")).exists() {
                    break;
                }
            },
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(orphaned);
    loop {
        tokio::select! {
            _ = &mut orphaned => break,
            accepted = listener.accept() => {
                let (stream, _) = accepted?;
                let (endpoint, store) = (endpoint.clone(), (*store).clone());
                tokio::spawn(async move {
                    if let Err(err) = control(stream, endpoint, store).await {
                        eprintln!("control: {err:#}");
                    }
                });
            }
            _ = tokio::signal::ctrl_c() => break,
        }
    }
    router.shutdown().await?;
    Ok(())
}

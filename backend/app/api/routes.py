import os
import json
import shutil
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse, StreamingResponse

from ..config import settings
from ..models import (
    ExtractionRequest,
    ExtractionResponse,
    JobStatus,
    AudioFileItem,
    SyncRequest,
    SyncResponse
)
from ..services.extractor import extractor_service, get_job, list_jobs, validate_youtube_url
from ..services.job_store import job_store, sanitize_error_message
from ..services.storage import storage_service

router = APIRouter(prefix="/api", tags=["EchoNode Audio Engine"])

WORKER_TOKEN_ENV = "ECHONODE_WORKER_TOKEN"
WORKER_TOKEN_HEADER = "X-EchoNode-Worker-Token"


def get_worker_url() -> str:
    """Dynamically get configured worker URL from environment."""
    return os.environ.get("WORKER_URL", "").strip().rstrip("/")


def get_worker_token() -> str:
    """Get the shared secret used for Vercel -> worker authentication."""
    return os.environ.get(WORKER_TOKEN_ENV, "").strip()

def is_cookie_configured() -> bool:
    """Return whether YouTube authentication cookies are present."""
    if os.environ.get("YTDLP_COOKIES") or os.environ.get("YTDLP_COOKIES_PATH"):
        return True
    cookie_candidates = [
        settings.downloads_dir.parent / "cookies.txt",
        settings.downloads_dir.parent / ".cookies"
    ]
    return any(p.exists() for p in cookie_candidates)

def is_po_token_provider_configured() -> bool:
    """Return whether a supported PO Token Provider plugin or service is configured."""
    try:
        import yt_dlp_plugins.extractor.getpot_bgutil
        return True
    except ImportError:
        return bool(os.environ.get("BGUTIL_BASE_URL") or os.environ.get("BGUTIL_SERVER_URL") or os.environ.get("POT_PROVIDER_URL"))

def is_persistent_storage() -> bool:
    """Return whether storage survives across process/container lifecycles."""
    if os.environ.get("STORAGE_BACKEND", "").lower() == "s3":
        return True
    is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("VERCEL_ENV"))
    return not is_serverless

def check_worker_health(timeout: float = 2.0) -> Tuple[bool, Optional[dict]]:
    """Probe the persistent worker health endpoint with a short timeout."""
    worker_url = get_worker_url()
    if not worker_url:
        return False, None
    try:
        token = get_worker_token()
        if not token:
            return False, None
        req = urllib.request.Request(
            f"{worker_url}/health",
            headers={
                "Accept": "application/json",
                "User-Agent": "EchoNode-API/1.1.0",
                WORKER_TOKEN_HEADER: token,
            },
            method="GET"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                return True, data
    except Exception:
        pass
    return False, None

def proxy_to_worker(method: str, path: str, payload: Optional[dict] = None) -> dict:
    """Forward an API request to the dedicated persistent extraction worker."""
    worker_url = get_worker_url()
    if not worker_url:
        raise HTTPException(status_code=503, detail="Extraction worker is not configured")

    target_url = f"{worker_url}{path}"
    token = get_worker_token()
    if not token:
        raise HTTPException(status_code=503, detail="Worker authentication secret is not configured")
    headers = {
        "Content-Type": "application/json",
        WORKER_TOKEN_HEADER: token,
    }
    data = json.dumps(payload).encode("utf-8") if payload else None

    req = urllib.request.Request(target_url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content) if content else {}
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_body)
            detail = err_json.get("detail", err_json.get("error", err_json.get("message", str(e))))
        except Exception:
            detail = err_body or str(e)
        raise HTTPException(status_code=e.code, detail=sanitize_error_message(detail))
    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Persistent extraction worker is currently unreachable: {sanitize_error_message(str(e.reason))}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=sanitize_error_message(str(e)))

@router.post("/extract", response_model=ExtractionResponse)
async def submit_extraction(request: ExtractionRequest):
    """Submit a YouTube URL to extract into an audio-only container."""
    url = request.url.strip()
    if not validate_youtube_url(url):
        raise HTTPException(
            status_code=400, 
            detail="Invalid or unsupported URL. Must be a valid HTTP/HTTPS link from youtube.com or youtu.be"
        )

    worker_url = get_worker_url()
    is_serverless = bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME") or os.environ.get("VERCEL_ENV"))
    extraction_mode = os.environ.get("EXTRACTION_MODE", "").strip().lower()

    # In production serverless or explicit worker mode, a persistent worker is required
    worker_required = (extraction_mode == "worker") or (is_serverless and extraction_mode != "local")

    if worker_required and not worker_url:
        raise HTTPException(status_code=503, detail="Extraction worker is not configured")

    # If configured with a standalone persistent worker, forward request to worker
    if worker_url:
        resp = proxy_to_worker("POST", "/api/extract", request.model_dump())
        return ExtractionResponse(**resp)

    job_id = await extractor_service.start_extraction_job(
        url=url,
        format_pref=request.format_preference or "m4a",
        custom_title=request.custom_title,
        wait_for_completion=is_serverless
    )

    job = get_job(job_id)
    status_str = job.status if (is_serverless and job) else "queued"
    msg = "Audio extracted and validated successfully" if status_str == "completed" else "Audio extraction job queued successfully"
    if status_str == "failed" and job and job.error:
        msg = f"Extraction failed: {job.error}"

    return ExtractionResponse(
        job_id=job_id,
        message=msg,
        status=status_str
    )

@router.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Retrieve progress and completion state of an extraction job."""
    worker_url = get_worker_url()
    if worker_url:
        resp = proxy_to_worker("GET", f"/api/jobs/{job_id}")
        return JobStatus(**resp)

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Extraction job not found")
    return job

@router.get("/jobs", response_model=List[JobStatus])
async def get_all_jobs():
    """Retrieve all jobs tracked in this persistent store."""
    worker_url = get_worker_url()
    if worker_url:
        resp = proxy_to_worker("GET", "/api/jobs")
        return [JobStatus(**item) for item in resp]

    return list_jobs()

@router.get("/files", response_model=List[AudioFileItem])
async def list_audio_files():
    """List all extracted audio files ready for SD transfer."""
    worker_url = get_worker_url()
    if worker_url:
        resp = proxy_to_worker("GET", "/api/files")
        return [AudioFileItem(**item) for item in resp]

    return storage_service.list_files()

@router.get("/download/{filename}")
async def download_file(filename: str):
    """Stream or redirect to download a specific audio track."""
    unquoted_name = urllib.parse.unquote(filename)
    worker_url = get_worker_url()

    if worker_url:
        token = get_worker_token()
        if not token:
            raise HTTPException(status_code=503, detail="Worker authentication secret is not configured")

        worker_download = f"{worker_url}/api/download/{urllib.parse.quote(unquoted_name)}"
        req = urllib.request.Request(
            worker_download,
            headers={
                WORKER_TOKEN_HEADER: token,
                "Accept": "audio/mp4, audio/mpeg, application/octet-stream",
                "User-Agent": "EchoNode-API/1.1.0",
            },
            method="GET",
        )
        try:
            upstream = urllib.request.urlopen(req, timeout=120)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            try:
                err_json = json.loads(err_body)
                detail = err_json.get("detail", err_json.get("error", err_json.get("message", str(e))))
            except Exception:
                detail = err_body or str(e)
            raise HTTPException(status_code=e.code, detail=sanitize_error_message(detail))
        except urllib.error.URLError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Persistent extraction worker is currently unreachable: {sanitize_error_message(str(e.reason))}"
            )

        media_type = upstream.headers.get_content_type() or (
            "audio/mp4" if Path(unquoted_name).suffix.lower() == ".m4a" else "audio/mpeg"
        )
        response_headers = {
            "Content-Disposition": f"attachment; filename*=UTF-8''{urllib.parse.quote(Path(unquoted_name).name)}"
        }
        content_length = upstream.headers.get("Content-Length")
        if content_length:
            response_headers["Content-Length"] = content_length

        def stream_worker_file():
            try:
                while True:
                    chunk = upstream.read(1024 * 1024)
                    if not chunk:
                        break
                    yield chunk
            finally:
                upstream.close()

        return StreamingResponse(stream_worker_file(), media_type=media_type, headers=response_headers)

    local_path = storage_service.get_file_path(unquoted_name)
    if local_path and local_path.exists():
        media_type = "audio/mp4" if local_path.suffix.lower() == ".m4a" else "audio/mpeg"
        return FileResponse(
            path=local_path,
            media_type=media_type,
            filename=local_path.name
        )

    # Check if object storage has a public/presigned URL
    if storage_service.file_exists(unquoted_name):
        return RedirectResponse(url=storage_service.get_download_url(unquoted_name))

    raise HTTPException(status_code=404, detail="Requested audio track not found")

@router.delete("/files/{filename}")
async def delete_file(filename: str):
    """Delete an audio track from storage."""
    unquoted_name = urllib.parse.unquote(filename)
    worker_url = get_worker_url()

    if worker_url:
        resp = proxy_to_worker("DELETE", f"/api/files/{urllib.parse.quote(unquoted_name)}")
        return resp

    if not storage_service.file_exists(unquoted_name):
        raise HTTPException(status_code=404, detail="File not found")

    storage_service.delete_file(unquoted_name)
    return {"message": f"Successfully deleted {unquoted_name}"}

@router.post("/sync-sd", response_model=SyncResponse)
async def sync_to_sd(req: SyncRequest):
    """Copy all staged audio files to a mounted MicroSD card path."""
    dest = Path(req.target_path)
    if not dest.exists() or not dest.is_dir():
        raise HTTPException(status_code=400, detail=f"Target path does not exist or is not a directory: {req.target_path}")

    copied = 0
    failed = 0
    valid_extensions = {".m4a", ".mp3", ".aac"}

    for item in storage_service.list_files():
        if f".{item.extension}" in valid_extensions:
            local_src = storage_service.get_file_path(item.filename)
            if local_src and local_src.exists():
                try:
                    dest_file = dest / item.filename
                    shutil.copy2(local_src, dest_file)
                    copied += 1
                except Exception:
                    failed += 1

    return SyncResponse(
        copied_count=copied,
        failed_count=failed,
        target_path=str(dest),
        message=f"Sync completed. Copied {copied} files ({failed} failed)."
    )

@router.get("/system-info")
async def system_info():
    """System capabilities, storage, and worker diagnostics (NON-SECRET)."""
    worker_url = get_worker_url()
    worker_configured = bool(worker_url)
    yt_dlp_v = "unknown"
    try:
        import yt_dlp
        yt_dlp_v = getattr(yt_dlp.version, "__version__", str(getattr(yt_dlp, "__version__", "unknown")))
    except Exception:
        pass

    files = storage_service.list_files()
    total_size = sum(f.size_bytes for f in files)

    if worker_configured:
        reachable, worker_details = check_worker_health(timeout=2.0)
        ffmpeg_avail = worker_details.get("ffmpeg_available", False) if (reachable and worker_details) else False
        cookie_cfg = worker_details.get("youtube_cookie_configured", False) if (reachable and worker_details) else is_cookie_configured()
        po_token_cfg = worker_details.get("youtube_po_token_provider_configured", False) if (reachable and worker_details) else is_po_token_provider_configured()
        storage_b = worker_details.get("storage_backend", "local") if (reachable and worker_details) else os.environ.get("STORAGE_BACKEND", "local")
        worker_yt_v = worker_details.get("yt_dlp_version", yt_dlp_v) if (reachable and worker_details) else yt_dlp_v
        worker_downloads = worker_details.get("downloads_dir", str(settings.downloads_dir)) if (reachable and worker_details) else str(settings.downloads_dir)

        return {
            "mode": "proxy_to_worker",
            "worker_configured": True,
            "worker_reachable": reachable,
            "ffmpeg_available": ffmpeg_avail,
            "storage_backend": storage_b,
            "youtube_cookie_configured": cookie_cfg,
            "youtube_po_token_provider_configured": po_token_cfg,
            "yt_dlp_version": worker_yt_v,
            "backend_version": settings.version,
            "downloads_dir": worker_downloads,
            "persistent_storage": is_persistent_storage(),
            "app": settings.app_name,
            "version": settings.version,
            "worker_status": "connected" if reachable else "disconnected",
            "worker_details": worker_details,
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "sd_format_standard": "FAT32 (<=32GB recommended)"
        }

    return {
        "mode": "standalone_local",
        "worker_configured": False,
        "worker_reachable": None,
        "ffmpeg_available": extractor_service.has_ffmpeg,
        "storage_backend": os.environ.get("STORAGE_BACKEND", "local"),
        "youtube_cookie_configured": is_cookie_configured(),
        "youtube_po_token_provider_configured": is_po_token_provider_configured(),
        "yt_dlp_version": yt_dlp_v,
        "backend_version": settings.version,
        "downloads_dir": str(settings.downloads_dir),
        "persistent_storage": is_persistent_storage(),
        "app": settings.app_name,
        "version": settings.version,
        "total_files": len(files),
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "sd_format_standard": "FAT32 (<=32GB recommended)"
    }

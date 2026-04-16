/**
 * Physiosis Frontend Logic
 * Handles video stream toggling and status polling.
 */

document.addEventListener('DOMContentLoaded', () => {
    
    const streamToggleBtn = document.getElementById('streamToggle');
    const videoStream = document.getElementById('videoStream');
    const videoOverlay = document.getElementById('videoOverlay');
    const engineStatus = document.getElementById('engineStatus');
    const pulseRing = document.querySelector('.pulse-ring');
    
    let isPlaying = true;
    
    // Toggle Stream (Pause/Resume on Frontend)
    streamToggleBtn.addEventListener('click', () => {
        isPlaying = !isPlaying;
        
        if (isPlaying) {
            // Resume stream by reloading the image source
            videoStream.src = "/video_feed?" + new Date().getTime();
            videoStream.style.opacity = '1';
            videoOverlay.classList.add('hidden');
            
            streamToggleBtn.innerHTML = '<i class="fa-solid fa-pause"></i> Pause';
            engineStatus.textContent = 'System Active';
            pulseRing.style.background = 'var(--success)';
            
        } else {
            // Pause stream by removing the source (stops MJPEG requesting)
            videoStream.src = "";
            videoStream.style.opacity = '0.3';
            videoOverlay.classList.remove('hidden');
            
            streamToggleBtn.innerHTML = '<i class="fa-solid fa-play"></i> Resume';
            engineStatus.textContent = 'Stream Paused';
            pulseRing.style.background = 'var(--warning)';
        }
    });
    
    // Periodically check if backend is alive
    setInterval(async () => {
        try {
            const res = await fetch('/status');
            const data = await res.json();
            
            if (!data.active && isPlaying) {
                engineStatus.textContent = 'Engine Offline';
                pulseRing.style.background = 'var(--danger)';
                videoOverlay.querySelector('p').textContent = "Connection Lost";
                videoOverlay.classList.remove('hidden');
            } else if (data.active && isPlaying) {
                engineStatus.textContent = 'System Active';
                pulseRing.style.background = 'var(--success)';
            }
        } catch (e) {
            engineStatus.textContent = 'Disconnected';
            pulseRing.style.background = 'var(--danger)';
            videoOverlay.querySelector('p').textContent = "Server Unreachable";
            videoOverlay.classList.remove('hidden');
        }
    }, 5000);
});

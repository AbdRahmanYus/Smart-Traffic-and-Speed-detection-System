// Plain vanilla JS. No framework, no build step - just fetch and polling.

let currentJobId = null;
let pollTimer = null;

const statusLine = document.getElementById("status-line");
const statusPill = document.getElementById("status-pill");
const progressFill = document.getElementById("progress-fill");
const depthImage = document.getElementById("depth-image");
const offendersBody = document.getElementById("offenders-body");
const videoCard = document.getElementById("video-card");
const resultVideo = document.getElementById("result-video");
const readoutMode = document.getElementById("readout-mode");
const readoutDetected = document.getElementById("readout-detected");
const readoutEffective = document.getElementById("readout-effective");
const sessionLabel = document.getElementById("session-label");

const lightButtons = document.querySelectorAll(".lens");
const autoButton = document.getElementById("auto-button");

function setActiveLightButton(state) {
  lightButtons.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.state === state);
  });
  autoButton.classList.toggle("active", state === "auto");
}

async function loadTrafficLightState() {
  const res = await fetch("/traffic-light");
  if (!res.ok) return;
  const data = await res.json();
  setActiveLightButton(data.override);
  readoutMode.textContent = data.override.toUpperCase();
}

async function setTrafficLightState(state) {
  const res = await fetch("/traffic-light", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ state }),
  });
  if (!res.ok) return;
  const data = await res.json();
  setActiveLightButton(data.override);
  readoutMode.textContent = data.override.toUpperCase();
}

async function replayScenario(scenario) {
  // While a job is still processing, this just changes the live
  // override (already handled by setTrafficLightState above). Once a
  // job is done, it instantly re-judges the cached crossing events
  // under the chosen scenario - no GPU work, no reprocessing. If the
  // job isn't done yet the server returns 409, which we ignore here.
  if (!currentJobId) return;
  const res = await fetch(`/jobs/${currentJobId}/recompute`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ scenario }),
  });
  if (res.ok) {
    renderOffenders(await res.json());
  }
}

lightButtons.forEach((btn) => {
  btn.addEventListener("click", () => {
    setTrafficLightState(btn.dataset.state);
    replayScenario(btn.dataset.state);
  });
});
autoButton.addEventListener("click", () => {
  setTrafficLightState("auto");
  replayScenario("auto");
});

function setStatusPill(status) {
  statusPill.className = "status-pill";
  if (status === "processing" || status === "postprocessing") {
    statusPill.classList.add("processing");
    statusPill.textContent = "Processing";
  } else if (status === "done") {
    statusPill.classList.add("done");
    statusPill.textContent = "Done";
  } else if (status === "error") {
    statusPill.classList.add("error");
    statusPill.textContent = "Error";
  } else {
    statusPill.classList.add("idle");
    statusPill.textContent = "Idle";
  }
}

function renderOffenders(offenders) {
  if (!offenders.length) {
    offendersBody.innerHTML = '<tr><td colspan="6" class="empty-row">No offenders recorded yet.</td></tr>';
    return;
  }
  offendersBody.innerHTML = offenders
    .map((o) => {
      const snapshot = o.snapshot_filename
        ? `<img src="/jobs/${currentJobId}/snapshots/${o.snapshot_filename}" alt="snapshot">`
        : "-";
      const speed = o.speed_kmh != null ? `${o.speed_kmh} km/h` : "-";
      return `<tr>
        <td>${snapshot}</td>
        <td>${o.track_id}</td>
        <td>${o.vehicle_type}</td>
        <td>${o.plate_number}</td>
        <td>${speed}</td>
        <td>${o.violation_type}</td>
      </tr>`;
    })
    .join("");
}

async function pollJob() {
  if (!currentJobId) return;

  const statusRes = await fetch(`/jobs/${currentJobId}`);
  if (statusRes.ok) {
    const job = await statusRes.json();
    progressFill.style.width = `${job.progress_percent}%`;
    statusLine.textContent =
      `Status: ${job.status} - frame ${job.current_frame}/${job.total_frames || "?"} ` +
      `- offenders so far: ${job.offender_count}`;
    setStatusPill(job.status);
    readoutDetected.textContent = job.traffic_light_detected.toUpperCase();
    readoutEffective.textContent = job.traffic_light_effective.toUpperCase();

    if (job.status === "error") {
      statusLine.textContent = `Error: ${job.error_message}`;
      stopPolling();
      return;
    }

    if (job.status === "done") {
      resultVideo.src = `/jobs/${currentJobId}/video`;
      videoCard.hidden = false;
      statusLine.textContent += " - done. Try the traffic light buttons to replay other scenarios.";
      stopPolling();
    }
  }

  const depthRes = await fetch(`/jobs/${currentJobId}/depth`);
  if (depthRes.ok) {
    depthImage.src = `/jobs/${currentJobId}/depth?t=${Date.now()}`;
    depthImage.classList.add("loaded");
  }

  const offendersRes = await fetch(`/jobs/${currentJobId}/offenders`);
  if (offendersRes.ok) {
    renderOffenders(await offendersRes.json());
  }
}

function startPolling() {
  stopPolling();
  pollTimer = setInterval(pollJob, 2000);
  pollJob();
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

document.getElementById("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("video-input");
  if (!fileInput.files.length) return;

  videoCard.hidden = true;
  offendersBody.innerHTML = '<tr><td colspan="6" class="empty-row">No offenders recorded yet.</td></tr>';
  depthImage.classList.remove("loaded");
  readoutDetected.textContent = "\u2014";
  readoutEffective.textContent = "\u2014";
  statusLine.textContent = "Uploading...";

  const formData = new FormData();
  formData.append("video", fileInput.files[0]);

  const res = await fetch("/jobs", { method: "POST", body: formData });
  if (!res.ok) {
    statusLine.textContent = "Upload failed. Is the server still loading its models?";
    return;
  }

  const data = await res.json();
  currentJobId = data.job_id;
  sessionLabel.textContent = data.session_label;
  sessionLabel.hidden = false;
  statusLine.textContent = "Processing started...";
  setStatusPill("processing");
  startPolling();
});

document.querySelectorAll(".accordion-header").forEach((header) => {
  header.addEventListener("click", () => {
    const expanded = header.getAttribute("aria-expanded") === "true";
    header.setAttribute("aria-expanded", String(!expanded));
    document.getElementById(header.dataset.target).classList.toggle("collapsed", expanded);
  });
});

loadTrafficLightState();

const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file");
const resultSec = document.getElementById("result");
const preview = document.getElementById("preview");
const loading = document.getElementById("loading");
const errorBox = document.getElementById("error");
const predBox = document.getElementById("prediction");

dropzone.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("keydown", (e) => { if (e.key === "Enter") fileInput.click(); });
dropzone.addEventListener("dragover", (e) => { e.preventDefault(); dropzone.classList.add("drag"); });
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag"));
dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

async function handleFile(file) {
  resultSec.hidden = false;
  predBox.hidden = true;
  errorBox.hidden = true;
  loading.hidden = false;
  preview.src = URL.createObjectURL(file);

  const form = new FormData();
  form.append("file", file);

  try {
    const res = await fetch("/predict", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Erreur du serveur");
    render(data);
  } catch (err) {
    errorBox.textContent = err.message;
    errorBox.hidden = false;
  } finally {
    loading.hidden = true;
  }
}

function pct(p) { return (p * 100).toFixed(1) + " %"; }

function render(data) {
  document.getElementById("topName").textContent = `${data.top.name} (${data.top.label})`;
  document.getElementById("topProb").textContent = `Confiance : ${pct(data.top.probability)}`;
  const bars = document.getElementById("bars");
  bars.innerHTML = "";
  data.predictions.forEach((p, i) => {
    const row = document.createElement("div");
    row.className = "bar-row" + (i === 0 ? " top" : "");
    row.innerHTML = `<span title="${p.name}">${p.label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(p.probability * 100).toFixed(1)}%"></div></div>
      <span class="bar-val">${pct(p.probability)}</span>`;
    bars.appendChild(row);
  });
  document.getElementById("meta").textContent = `${data.model} · inférence : ${data.inference_ms} ms`;
  predBox.hidden = false;
}

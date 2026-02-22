const fileInput = document.getElementById("image-file");
const urlInput = document.getElementById("image-url");
const preview = document.getElementById("preview");
const previewEmpty = document.getElementById("preview-empty");
const analyzeBtn = document.getElementById("analyze-btn");
const resetBtn = document.getElementById("reset-btn");
const statusEl = document.getElementById("status");
const resultPanel = document.getElementById("result-panel");
const noFacePanel = document.getElementById("noface-panel");
const labelEl = document.getElementById("label");
const confidenceEl = document.getElementById("confidence");
const barEl = document.getElementById("bar");
const reasonsEl = document.getElementById("reasons");

const allowedExtensions = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"];

function setStatus(text, type) {
  statusEl.textContent = text;
  statusEl.className = `status status-${type}`;
}

function clearResult() {
  resultPanel.hidden = true;
  noFacePanel.hidden = true;
  reasonsEl.innerHTML = "";
  barEl.style.width = "0%";
}

function showPreview(src) {
  preview.src = src;
  preview.hidden = false;
  previewEmpty.hidden = true;
}

function clearPreview() {
  preview.src = "";
  preview.hidden = true;
  previewEmpty.hidden = false;
}

function extensionFromName(name) {
  const parts = name.toLowerCase().split(".");
  return parts.length > 1 ? parts.pop() : "";
}

function isAllowedFormat(nameOrUrl) {
  const ext = extensionFromName(nameOrUrl);
  return allowedExtensions.includes(ext);
}

function mockAnalyze(inputRef) {
  const ref = inputRef.toLowerCase();

  if (ref.includes("noface") || ref.includes("landscape") || ref.includes("cat")) {
    return {
      hasFace: false,
    };
  }

  if (ref.includes("ai") || ref.includes("generated") || ref.includes("midjourney")) {
    return {
      hasFace: true,
      label: "AI-generated",
      confidence: 88,
      reasons: [
        "Skin texture appears unusually uniform in several regions",
        "Fine background details show synthetic blending artifacts",
        "Specular highlights around the eyes appear physically inconsistent",
      ],
    };
  }

  return {
    hasFace: true,
    label: "Real",
    confidence: 82,
    reasons: [
      "Skin pores and micro-variations look naturally distributed",
      "Lighting and shadow transitions align with realistic face geometry",
      "No strong synthetic boundary artifacts are visible",
    ],
  };
}

fileInput.addEventListener("change", () => {
  clearResult();
  setStatus("File selected. Ready to analyze.", "idle");
  const file = fileInput.files?.[0];
  if (!file) {
    clearPreview();
    return;
  }
  if (!isAllowedFormat(file.name)) {
    clearPreview();
    setStatus("Unsupported file format. Please use JPG, JPEG, PNG, WEBP, BMP, or TIFF.", "error");
    fileInput.value = "";
    return;
  }
  showPreview(URL.createObjectURL(file));
});

urlInput.addEventListener("input", () => {
  clearResult();
  const url = urlInput.value.trim();
  if (!url) {
    clearPreview();
    setStatus("Waiting for input...", "idle");
    return;
  }

  try {
    const parsed = new URL(url);
    if (!isAllowedFormat(parsed.pathname)) {
      setStatus("Unsupported URL format. Use an image link ending in JPG, JPEG, PNG, WEBP, BMP, or TIFF.", "error");
      clearPreview();
      return;
    }
    showPreview(url);
    setStatus("Image URL set. Ready to analyze.", "idle");
  } catch {
    clearPreview();
    setStatus("Please enter a valid image URL.", "error");
  }
});

analyzeBtn.addEventListener("click", async () => {
  clearResult();
  const file = fileInput.files?.[0];
  const url = urlInput.value.trim();

  if (!file && !url) {
    setStatus("Please upload an image or provide an image URL.", "error");
    return;
  }

  let inputRef = "";
  if (file) {
    inputRef = file.name;
  } else {
    try {
      const parsed = new URL(url);
      inputRef = parsed.pathname;
    } catch {
      setStatus("Please enter a valid image URL.", "error");
      return;
    }
  }

  setStatus("Analyzing image (mock)...", "working");
  await new Promise((resolve) => setTimeout(resolve, 600));

  const result = mockAnalyze(inputRef);

  if (!result.hasFace) {
    noFacePanel.hidden = false;
    setStatus("Analysis complete: no face detected.", "error");
    return;
  }

  resultPanel.hidden = false;
  labelEl.textContent = result.label;
  confidenceEl.textContent = `${result.confidence}%`;
  barEl.style.width = `${result.confidence}%`;

  if (result.confidence >= 75) {
    barEl.style.background = "var(--ok)";
  } else if (result.confidence >= 40) {
    barEl.style.background = "var(--warn)";
  } else {
    barEl.style.background = "var(--bad)";
  }

  result.reasons.forEach((reason) => {
    const li = document.createElement("li");
    li.textContent = reason;
    reasonsEl.appendChild(li);
  });

  if (result.confidence >= 40 && result.confidence <= 60) {
    setStatus("Analysis complete: uncertain result. Try a clearer close-up face image.", "working");
  } else {
    setStatus("Analysis complete.", "done");
  }
});

resetBtn.addEventListener("click", () => {
  fileInput.value = "";
  urlInput.value = "";
  clearPreview();
  clearResult();
  setStatus("Waiting for input...", "idle");
});

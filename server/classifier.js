import fs from "node:fs/promises";
import path from "node:path";
import sharp from "sharp";

const modelPath = path.resolve(process.cwd(), "server/model/classifier.json");

let cachedModel = null;

export const FEATURE_NAMES = [
  "rMean",
  "gMean",
  "bMean",
  "rStd",
  "gStd",
  "bStd",
  "grayMean",
  "grayStd",
  "edgeMean",
  "edgeStd",
  "laplaceMean",
  "laplaceStd",
  "hfEnergy",
  "blockinessH",
  "blockinessV",
  "centerBrightnessDelta",
  "centerContrastDelta",
  "rgCorr",
  "rbCorr",
  "gbCorr",
  "histR0",
  "histR1",
  "histR2",
  "histR3",
  "histR4",
  "histR5",
  "histR6",
  "histR7",
  "histG0",
  "histG1",
  "histG2",
  "histG3",
  "histG4",
  "histG5",
  "histG6",
  "histG7",
  "histB0",
  "histB1",
  "histB2",
  "histB3",
  "histB4",
  "histB5",
  "histB6",
  "histB7",
  "lbp0",
  "lbp1",
  "lbp2",
  "lbp3",
  "lbp4",
  "lbp5",
  "lbp6",
  "lbp7",
  "lbp8",
  "lbp9",
  "lbp10",
  "lbp11",
  "lbp12",
  "lbp13",
  "lbp14",
  "lbp15",
];

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function safeStd(v) {
  return v > 1e-8 ? v : 1;
}

function sobelAndLaplacianFeatures(gray, width, height) {
  let edgeSum = 0;
  let edgeSq = 0;
  let lapSum = 0;
  let lapSq = 0;
  let hfSum = 0;
  let count = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const idx = y * width + x;
      const tl = gray[idx - width - 1];
      const tc = gray[idx - width];
      const tr = gray[idx - width + 1];
      const ml = gray[idx - 1];
      const mc = gray[idx];
      const mr = gray[idx + 1];
      const bl = gray[idx + width - 1];
      const bc = gray[idx + width];
      const br = gray[idx + width + 1];

      const gx = -tl - 2 * ml - bl + tr + 2 * mr + br;
      const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
      const mag = Math.sqrt(gx * gx + gy * gy);

      const lap = Math.abs(4 * mc - tc - ml - mr - bc);

      edgeSum += mag;
      edgeSq += mag * mag;
      lapSum += lap;
      lapSq += lap * lap;
      hfSum += Math.abs(mc - (tl + tc + tr + ml + mr + bl + bc + br) / 8);
      count += 1;
    }
  }

  const edgeMean = count ? edgeSum / count : 0;
  const edgeStd = count ? Math.sqrt(Math.max(0, edgeSq / count - edgeMean * edgeMean)) : 0;
  const lapMean = count ? lapSum / count : 0;
  const lapStd = count ? Math.sqrt(Math.max(0, lapSq / count - lapMean * lapMean)) : 0;
  const hfEnergy = count ? hfSum / count : 0;

  return [edgeMean, edgeStd, lapMean, lapStd, hfEnergy];
}

function correlation(a, b, meanA, meanB, stdA, stdB) {
  if (stdA < 1e-8 || stdB < 1e-8) {
    return 0;
  }

  let cov = 0;
  const n = a.length;
  for (let i = 0; i < n; i += 1) {
    cov += (a[i] - meanA) * (b[i] - meanB);
  }
  cov /= n;
  return cov / (stdA * stdB);
}

function blockiness(gray, width, height) {
  let hDiff = 0;
  let hCount = 0;
  for (let y = 0; y < height; y += 1) {
    for (let x = 8; x < width; x += 8) {
      const idx = y * width + x;
      hDiff += Math.abs(gray[idx] - gray[idx - 1]);
      hCount += 1;
    }
  }

  let vDiff = 0;
  let vCount = 0;
  for (let y = 8; y < height; y += 8) {
    for (let x = 0; x < width; x += 1) {
      const idx = y * width + x;
      vDiff += Math.abs(gray[idx] - gray[idx - width]);
      vCount += 1;
    }
  }

  return [hCount ? hDiff / hCount : 0, vCount ? vDiff / vCount : 0];
}

function normalizedHistogram(values, bins = 8) {
  const hist = Array(bins).fill(0);
  for (let i = 0; i < values.length; i += 1) {
    const v = values[i];
    const idx = Math.max(0, Math.min(bins - 1, Math.floor(v * bins)));
    hist[idx] += 1;
  }
  const inv = 1 / values.length;
  for (let i = 0; i < bins; i += 1) {
    hist[i] *= inv;
  }
  return hist;
}

function lbpHistogram(gray, width, height) {
  const bins = Array(16).fill(0);
  let count = 0;

  for (let y = 1; y < height - 1; y += 1) {
    for (let x = 1; x < width - 1; x += 1) {
      const idx = y * width + x;
      const c = gray[idx];
      let code = 0;

      const n0 = gray[idx - width - 1] >= c ? 1 : 0;
      const n1 = gray[idx - width] >= c ? 1 : 0;
      const n2 = gray[idx - width + 1] >= c ? 1 : 0;
      const n3 = gray[idx + 1] >= c ? 1 : 0;
      const n4 = gray[idx + width + 1] >= c ? 1 : 0;
      const n5 = gray[idx + width] >= c ? 1 : 0;
      const n6 = gray[idx + width - 1] >= c ? 1 : 0;
      const n7 = gray[idx - 1] >= c ? 1 : 0;

      code = n0 | (n1 << 1) | (n2 << 2) | (n3 << 3) | (n4 << 4) | (n5 << 5) | (n6 << 6) | (n7 << 7);
      bins[code >> 4] += 1;
      count += 1;
    }
  }

  if (count === 0) return bins;
  const inv = 1 / count;
  for (let i = 0; i < bins.length; i += 1) {
    bins[i] *= inv;
  }
  return bins;
}

function centerStats(gray, width, height) {
  const x0 = Math.floor(width * 0.25);
  const x1 = Math.floor(width * 0.75);
  const y0 = Math.floor(height * 0.25);
  const y1 = Math.floor(height * 0.75);

  let cSum = 0;
  let cSq = 0;
  let cN = 0;
  let oSum = 0;
  let oSq = 0;
  let oN = 0;

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const v = gray[y * width + x];
      const inCenter = x >= x0 && x < x1 && y >= y0 && y < y1;
      if (inCenter) {
        cSum += v;
        cSq += v * v;
        cN += 1;
      } else {
        oSum += v;
        oSq += v * v;
        oN += 1;
      }
    }
  }

  const cMean = cN ? cSum / cN : 0;
  const oMean = oN ? oSum / oN : 0;
  const cStd = cN ? Math.sqrt(Math.max(0, cSq / cN - cMean * cMean)) : 0;
  const oStd = oN ? Math.sqrt(Math.max(0, oSq / oN - oMean * oMean)) : 0;

  return [cMean - oMean, cStd - oStd];
}

export function applyFeatureTransform(features, model) {
  const normalized = features.map((value, idx) => (value - model.featureMean[idx]) / safeStd(model.featureStd[idx]));

  if (model.transform === "zscore_poly2") {
    const squared = normalized.map((v) => v * v);
    return [...normalized, ...squared];
  }

  return normalized;
}

export async function extractFeaturesFromBuffer(imageBuffer) {
  const { data, info } = await sharp(imageBuffer)
    .resize(160, 160, { fit: "fill" })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixelCount = info.width * info.height;
  const gray = new Float32Array(pixelCount);
  const rs = new Float32Array(pixelCount);
  const gs = new Float32Array(pixelCount);
  const bs = new Float32Array(pixelCount);

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let rSq = 0;
  let gSq = 0;
  let bSq = 0;
  let graySum = 0;
  let graySq = 0;

  for (let i = 0, p = 0; i < data.length; i += 3, p += 1) {
    const r = data[i] / 255;
    const g = data[i + 1] / 255;
    const b = data[i + 2] / 255;

    rs[p] = r;
    gs[p] = g;
    bs[p] = b;

    rSum += r;
    gSum += g;
    bSum += b;
    rSq += r * r;
    gSq += g * g;
    bSq += b * b;

    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    gray[p] = y;
    graySum += y;
    graySq += y * y;
  }

  const rMean = rSum / pixelCount;
  const gMean = gSum / pixelCount;
  const bMean = bSum / pixelCount;
  const rStd = Math.sqrt(Math.max(0, rSq / pixelCount - rMean * rMean));
  const gStd = Math.sqrt(Math.max(0, gSq / pixelCount - gMean * gMean));
  const bStd = Math.sqrt(Math.max(0, bSq / pixelCount - bMean * bMean));

  const grayMean = graySum / pixelCount;
  const grayStd = Math.sqrt(Math.max(0, graySq / pixelCount - grayMean * grayMean));

  const [edgeMean, edgeStd, laplaceMean, laplaceStd, hfEnergy] = sobelAndLaplacianFeatures(gray, info.width, info.height);
  const [blockinessH, blockinessV] = blockiness(gray, info.width, info.height);
  const [centerBrightnessDelta, centerContrastDelta] = centerStats(gray, info.width, info.height);

  const rgCorr = correlation(rs, gs, rMean, gMean, rStd, gStd);
  const rbCorr = correlation(rs, bs, rMean, bMean, rStd, bStd);
  const gbCorr = correlation(gs, bs, gMean, bMean, gStd, bStd);

  const rHist = normalizedHistogram(rs, 8);
  const gHist = normalizedHistogram(gs, 8);
  const bHist = normalizedHistogram(bs, 8);
  const lbpHist = lbpHistogram(gray, info.width, info.height);

  return [
    rMean,
    gMean,
    bMean,
    rStd,
    gStd,
    bStd,
    grayMean,
    grayStd,
    edgeMean,
    edgeStd,
    laplaceMean,
    laplaceStd,
    hfEnergy,
    blockinessH,
    blockinessV,
    centerBrightnessDelta,
    centerContrastDelta,
    rgCorr,
    rbCorr,
    gbCorr,
    ...rHist,
    ...gHist,
    ...bHist,
    ...lbpHist,
  ];
}

export async function loadClassifierModel() {
  if (cachedModel) {
    return cachedModel;
  }

  try {
    const raw = await fs.readFile(modelPath, "utf8");
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed.weights) || !Array.isArray(parsed.featureMean) || !Array.isArray(parsed.featureStd)) {
      throw new Error("Invalid model file format");
    }
    cachedModel = parsed;
    return cachedModel;
  } catch {
    return null;
  }
}

export async function classifyImageBuffer(imageBuffer, inputRef = "") {
  const model = await loadClassifierModel();
  if (!model) {
    return null;
  }

  const features = await extractFeaturesFromBuffer(imageBuffer);
  const transformed = applyFeatureTransform(features, model);

  let logit = model.bias;
  for (let i = 0; i < transformed.length; i += 1) {
    logit += transformed[i] * model.weights[i];
  }

  const pReal = sigmoid(logit);
  const pFake = 1 - pReal;
  const aiConfidence = Math.max(0, Math.min(100, Math.round(pFake * 100)));

  const reasons =
    aiConfidence >= 50
      ? [
          "Model found synthetic-like texture, local pattern, and frequency cues.",
          "Color histogram and edge statistics diverge from typical real-face profiles.",
          "Prediction uses archive-trained forensic feature patterns.",
        ]
      : [
          "Model found natural texture and local pattern consistency.",
          "Color histogram and edge statistics align with typical real-face profiles.",
          "Prediction uses archive-trained forensic feature patterns.",
        ];

  return {
    aiConfidence,
    reasons,
    source: "archive-trained-poly-logistic",
    inputRef,
  };
}

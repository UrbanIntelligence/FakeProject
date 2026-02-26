import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const checkpointPath = path.resolve(process.cwd(), "ml/artifacts/best_resnet18.pt");
const inferScriptPath = path.resolve(process.cwd(), "ml/infer_transfer.py");
const enableTransferModel = process.env.ENABLE_TRANSFER_MODEL === "1";

let checkedAvailability = false;
let transferAvailable = false;

async function checkTransferAvailability() {
  if (checkedAvailability) {
    return transferAvailable;
  }

  checkedAvailability = true;
  if (!enableTransferModel) {
    transferAvailable = false;
    return transferAvailable;
  }

  try {
    await fs.access(checkpointPath);
    await fs.access(inferScriptPath);
    transferAvailable = true;
  } catch {
    transferAvailable = false;
  }

  return transferAvailable;
}

export async function classifyWithTransferModel(imageBuffer) {
  const available = await checkTransferAvailability();
  if (!available) {
    return null;
  }

  const tmpDir = await fs.mkdtemp(path.join(os.tmpdir(), "transfer-infer-"));
  const imagePath = path.join(tmpDir, "input.jpg");

  try {
    await fs.writeFile(imagePath, imageBuffer);

    const { stdout } = await execFileAsync(
      "python3",
      [inferScriptPath, "--image", imagePath, "--checkpoint", checkpointPath],
      {
        timeout: 20000,
        maxBuffer: 1024 * 1024,
      },
    );

    const output = JSON.parse(stdout.trim());
    if (!output.ok || typeof output.aiConfidence !== "number") {
      return null;
    }

    return {
      aiConfidence: output.aiConfidence,
      reasons: Array.isArray(output.reasons) ? output.reasons : [],
      source: output.source || "transfer-resnet18",
    };
  } catch {
    return null;
  } finally {
    await fs.rm(tmpDir, { recursive: true, force: true });
  }
}

export async function getTransferModelStatus() {
  const available = await checkTransferAvailability();
  return {
    enabled: enableTransferModel,
    available,
    checkpointPath,
    inferScriptPath,
  };
}

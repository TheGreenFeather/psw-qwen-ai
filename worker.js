import * as ort from "onnxruntime-web";
// import { Tokenizer } from "tokenizers";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// Model and Tokenizer URLs
const MODEL_URL =
  "https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main/onnx/model_int8.onnx";
const TOKENIZER_URL =
  "https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json";

let tokenizerConfig = null;
let session = null;

async function loadTokenizerConfig() {
  const response = await fetch(TOKENIZER_URL);
  if (!response.ok) {
    throw new Error("Failed to fetch tokenizer configuration");
  }
  tokenizerConfig = await response.json();
}

async function loadModel() {
  session = await ort.InferenceSession.create(MODEL_URL);
}

loadModel().then(() => console.log("Model loaded successfully"));
loadTokenizerConfig().then(() => console.log("Tokenizer loaded successfully"));

function tokenize(text) {
  if (
    !tokenizerConfig ||
    !tokenizerConfig.model ||
    !tokenizerConfig.model.vocab
  ) {
    throw new Error("Tokenizer not properly loaded");
  }
  const vocab = tokenizerConfig.model.vocab;
  // Define unknown token id.
  const unkId = vocab["[UNK]"] || 0;
  // Split text by whitespace.
  const tokens = text.split(/\s+/);
  return tokens.map((token) =>
    vocab[token] !== undefined ? vocab[token] : unkId
  );
}

function detokenize(tokenIds) {
  if (
    !tokenizerConfig ||
    !tokenizerConfig.model ||
    !tokenizerConfig.model.vocab
  ) {
    throw new Error("Tokenizer not properly loaded");
  }
  const vocab = tokenizerConfig.model.vocab;
  const idToToken = Object.fromEntries(
    Object.entries(vocab).map(([token, id]) => [id, token])
  );
  return tokenIds.map((id) => idToToken[id] || "[UNK]").join(" ");
}

async function generateText(prompt) {
  const tokenIds = await tokenize(prompt);
  const inputTensor = new ort.Tensor("int8", Int32Array.from(tokenIds), [
    1,
    tokenIds.length,
  ]);
  const feeds = { input_ids: inputTensor };
  const results = await session.run(feeds);
  const outputTensor = results.output_ids;
  const outputTokenIds = outputTensor.data;
  const generatedText = await detokenize(outputTokenIds);
  return generatedText;
}

// Cloudflare Worker Event Listener
addEventListener("fetch", (event) => {
  event.respondWith(handleRequest(event.request));
});

/**
 * Handles incoming requests. Expects a POST with JSON { "prompt": "..." }.
 */
async function handleRequest(request) {
  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type, Authorization, orgin, request",
      },
    });
  }

  if (
    request.method === "POST" &&
    request.headers.get("Content-Type") === "application/json" &&
    request.body &&
    request.url ===
      "https://psw-qwen-ai.nguyenvuong17102008.workers.dev/qwen2-5/ask"
  ) {
    try {
      const { message } = await request.json();
      const generatedText = await generateText(message);
      return new Response(JSON.stringify({ message: generatedText }), {
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
      });
    }
  }
  return new Response("Invalid request message", { status: 400 });
}

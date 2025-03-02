import * as ort from "onnxruntime-web";
// import { Tokenizer } from "tokenizers";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// Model and Tokenizer URLs
const MODEL_URL =
  "https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main/onnx/model_int8.onnx";
const TOKENIZER_URL =
  "https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json";

// let tokenizerConfig = null;
// let session = null;

// console.log("Started");

// async function loadModel() {
  
// }

// loadModel().then(() => console.log("Model loaded successfully"));
// loadTokenizerConfig().then(() => console.log("Tokenizer loaded successfully"));

async function tokenize(text) {
  const response = await fetch(TOKENIZER_URL);
  const tokenizerConfig = await response.json();
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

async function detokenize(tokenIds) {
  const response = await fetch(TOKENIZER_URL);
  const tokenizerConfig = await response.json();
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
  const session = await ort.InferenceSession.create(MODEL_URL);
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

const corsHeaders = {
  "Access-Control-Allow-Methods": "GET,HEAD,POST,OPTIONS", // Allowed methods. Others could be GET, PUT, DELETE etc.
  "Access-Control-Allow-Origin": "*", // This is URLs that are allowed to access the server. * is the wildcard character meaning any URL can.
  "Access-Control-Max-Age": "86400",
};

const API_ENDPOINT = "/qwen/ask";

async function handleRequest(request) {
  const url = new URL(request.url);
  if (url.pathname.startsWith(API_ENDPOINT)) {
    if (request.method === "OPTIONS") {
      if (
        request.headers.get("Origin") !== null &&
        request.headers.get("Access-Control-Request-Method") !== null &&
        request.headers.get("Access-Control-Request-Headers") !== null
      ) {
        // Handle CORS pre-flight request.
        return new Response(null, {
          headers: {
            ...corsHeaders,
            "Access-Control-Allow-Headers": request.headers.get(
              "Access-Control-Request-Headers",
            ),
          }
        });
      } else {
        // Handle standard OPTIONS request.
        return new Response(null, {
          headers: {
            ...corsHeaders,
            Allow: "GET,HEAD,POST,OPTIONS",
          },
        });
      }
    } else if (request.method === "POST") {
      return doTheWork(request);
    } else {
      return new Response("Method not allowed", {
        status: 405,
        headers: corsHeaders,
      });
    }
  }
}

async function doTheWork(request) {
  // Parse the request body to get the parameters
  
  const requestBody = await request.json();
  const message = requestBody.message;

  const generatedText = await generateText(message);

  //do the work here

  return new Response(JSON.stringify({ message: generatedText }), {
    headers: {
      "Vary": "Accept-Encoding",
      "Origin": "*",
      "Content-type": "application/json",
      ...corsHeaders, //uses the spread operator to include the CORS headers.
    },
  });
}

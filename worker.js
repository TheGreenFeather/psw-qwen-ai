import * as ort from "onnxruntime-web";
// import { Tokenizer } from "tokenizers";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/";

// Model and Tokenizer URLs
const MODEL_URL =
  "https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main/onnx/model_int8.onnx";
const TOKENIZER_URL =
  "https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main/tokenizer.json";

// Load tokenizer

// console.log();

throw new Error(process.platform);

// const tokenizerPromise = await Tokenizer.fromFile(TOKENIZER_URL);


// const tokenizer = await tokenizerPromise;


/**
 * Runs inference on the model for a given prompt.
 */
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

/**
 * Tokenizes the given text.
 */
async function tokenize(text) {
  // const encoding = tokenizer.encode(text);
  return [1];
}

/**
 * Detokenizes an array of token IDs back into a string.
 */
async function detokenize(tokenIds) {
  // const decoded = tokenizer.decode(Array.from(tokenIds));
  return '10';
}

// Cloudflare Worker Event Listener
addEventListener("fetch", (event) => {
  event.respondWith(handleRequest(event.request));
});

/**
 * Handles incoming requests. Expects a POST with JSON { "prompt": "..." }.
 */
async function handleRequest(request) {
  if (
    request.method === "POST" &&
    request.headers.get("Content-Type") === "application/json" &&
    request.body &&
    request.url === "https://psw-qwen-ai.nguyenvuong17102008.workers.dev/qwen2-5/ask"
  ) {
    try {
      const { message } = await request.json();
      const generatedText = await generateText(message);
      return new Response(JSON.stringify({ generatedText }), {
        headers: { "Content-Type": "application/json" },
      });
    } catch (error) {
      return new Response(JSON.stringify({ error: error.message }), {
        status: 500,
      });
    }
  }
  return new Response("Invalid request message", { status: 400 });
}

//  The code above is a Cloudflare Worker script that runs the ONNX model on the browser. It uses the  onnxruntime-web  library to load the model and run inference. The script also loads the tokenizer and uses it to tokenize and detokenize text. 
//  The script listens for incoming requests and expects a POST request with a JSON body containing a  message  field. It then generates text based on the given message and returns the generated text in the response. 
//  3. Deploy the Worker 
//  To deploy the Worker, you need to use the  Wrangler CLI. If you haven’t installed the Wrangler CLI yet, you can install it by running the following command: 
//  npm install -g @cloudflare/wrangler

//  Next, you need to create a new Worker project using the following command: 
//  wrangler generate qwen-ai
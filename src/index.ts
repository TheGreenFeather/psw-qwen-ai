import { pipeline } from "@huggingface/transformers";
import { Hono } from "hono";
import { HTTPException } from "hono/http-exception";

const app = new Hono();

const model_name = "onnx-community/Qwen2.5-1.5B-Instruct";

console.log(`Loading model: ${model_name}`);

const Qwen = await pipeline("text-generation", model_name, {
  // device: "cpu",
  dtype: "q4",
});

console.log("Model loaded");

app.post("/qwen/ask", async (c) => {
  const data = await c.req.parseBody();

  if (!data.message) {
    throw new HTTPException(400, { message: "Invalid request body" });
  }

  console.log("Received request with a message:", data.message);

  const messages = [{ role: "user", content: data.message }];

  const response = await Qwen(messages, { max_new_tokens: 512 });

  const response_message = response[0].generated_text.at(-1).content;

  console.log("Response generated:", response_message);

  console.log("Response sent");

  return c.json({ message: response_message });
});

export default app;

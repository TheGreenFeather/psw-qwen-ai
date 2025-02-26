import { pipeline } from "@huggingface/transformers";
import bodyParser from "body-parser";
import express from "express";
import cors from "cors";

const model_name = "onnx-community/Qwen2.5-1.5B-Instruct";

console.log(`Loading model: ${model_name}`);

const Qwen = await pipeline("text-generation", model_name, {
  // device: "cpu",
  dtype: "q4",
});

console.log("Model loaded");

const port = process.env.PORT || 8000;
const app = express();

app.use(express.json());
app.use(bodyParser.json());

app.use(
  cors({
    origin: "*",
    methods: ["POST"],
  })
);

app.use((req, res, next) => {
  res.setHeader("Content-Type", "application/json");
  next();
});

app.post("/qwen/ask", async function (req, res) {

  const data = req.body;

  if (!data.message) {
    return res.status(400).json({ error: "Invalid request body" });
  }

  console.log("Received request with a message:", data.message);

  const messages = [{ role: "user", content: data.message }];

  const response = await Qwen(messages, { max_new_tokens: 512 });

  const response_message =
    response[0].generated_text.at(-1).content;

  console.log("Response generated:", response_message);
    
  res.status(200).json({ message: response_message });
  
  console.log("Response sent");
});

const server = app.listen(port || 8000, function () {
  console.log(`Server listening on port ${port}`);
});

server.keepAliveTimeout = 1800 * 1000;
server.headersTimeout = 120 * 1000;

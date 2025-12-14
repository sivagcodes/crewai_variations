// sampleNodeOpenAI.js
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: "sk-proj---",
});

async function run() {
  try {
    const result = await openai.responses.create({
      model: "gpt-4o-mini",      // use a currently‑available model
      input: "write a haiku about AI",
      store: true,
    });
    console.log(result.output_text);
  } catch (err) {
    console.error("OpenAI error:", err);
  }
}

run();
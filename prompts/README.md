# LLM Prompts

This directory contains the prompts used by the Kimi LLM for RAG (Retrieval-Augmented Generation).

## Files

### `system_prompt.txt`
The system prompt that defines the LLM's role and instructions. This sets the behavior and tone of the assistant.

**Customize this to:**
- Change how the assistant interprets passages
- Adjust citation style
- Modify the level of detail in responses
- Change the tone (formal, casual, technical, etc.)

### `user_prompt_template.txt`
The template for the user query that gets sent to the LLM. Uses Python string formatting with placeholders:
- `{question}` - The user's question
- `{context}` - The retrieved passages from the book

**Customize this to:**
- Change how passages are presented to the LLM
- Add additional instructions per query
- Modify the format of the answer request

## Usage

The prompts are automatically loaded by `examples/novel_rag_with_llm.py`. Simply edit these text files and your changes will take effect on the next run.

No code changes needed - just edit the `.txt` files!

## Tips for Prompt Engineering

1. **Be specific**: Clear instructions lead to better outputs
2. **Use examples**: Show the LLM what you want (few-shot learning)
3. **Control temperature**: Lower = more focused, Higher = more creative (set in code)
4. **Iterate**: Test different phrasings to see what works best
5. **Length control**: Add token/word limits in the prompt itself

## Example Modifications

### More concise answers:
Add to system_prompt.txt:
```
- Keep answers under 3 paragraphs
- Use bullet points for lists
```

### Citation format:
Change in system_prompt.txt:
```
- Cite sources as (p. 123) instead of "Page 123"
```

### Add reasoning:
Add to user_prompt_template.txt:
```
First explain your reasoning, then provide the final answer.
```

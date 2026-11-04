# Profiles

A profile is a directory. The only required file is `constitution.md`: the
persona in plain language, always placed first in the system prompt.

```
profiles/
  stranger/
    constitution.md     who the Stranger is, how it speaks, what it will not do
    profile.toml        display name, speaker tag, preamble, optional context
  assistant/
    constitution.md
    profile.toml
```

## profile.toml

| Key | Default | Meaning |
|---|---|---|
| `display_name` | the directory name, title-cased | label on the page and in the terminal |
| `speaker_tag` | `display_name` upper-cased | the agent's tag in the conversation history |
| `preamble` | none | a paragraph after the constitution; `{name}` is replaced with the user's name |
| `context_file` | none | a text file, relative to the profile directory, placed after the preamble verbatim |
| `context_label` | `CONTEXT` | the heading above the context block |
| `stop_words` | `[]` | extra stop sequences for the model, e.g. other speakers' tags |

The system prompt is built as:

```
<constitution>

<preamble with {name} filled in>

<context_label>:
---
<context file>
---
```

## Writing a constitution

What worked for the Stranger, and transfers to other personas:

- Say how the persona speaks (register, sentence length, no markdown: the
  text is read aloud).
- Give a sentence budget. Small models run on; "two or three sentences at
  most" is respected far better than "be concise".
- List limits positively and concretely: speak only as yourself, no
  labels, no stage directions, do not invent people or events, say when
  you do not know.
- Keep the constitution and the context separate. Facts the persona should
  know go in the context file, not in the constitution.

## Using a profile

```bash
python -m stranger profiles                              # list what is found under profiles/
python -m stranger serve --profile assistant
python -m stranger serve --profile ~/personas/gardener   # any directory with a constitution.md
```

The trigger word for the `named` reply rule defaults to the last word of
the display name ("The Stranger" -> "Stranger"); set `--trigger-name` to
change it.

---
name: python-type-checking
description: Project type checking. Use after any Python modification to check validity.
---

# Type checking

Run the project's type checks with pyright to validate code changes.

## When to use

Run the check after any Python modification.

## Running the tests

Run the full check:

```bash
uv run pyright
```

When you modify a specific feature, you can run only the check for that feature to get faster feedback:

```bash
uv run pyright <file_name>.py
```

Replace `<file_name>` with the file covering the feature you changed. Run the relevant subset first while iterating, then run the full suite before finishing to make sure nothing else broke.
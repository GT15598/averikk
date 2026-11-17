# Contributing

## Workflow

1. Create a branch from `main`:

   ```bash
   git checkout main
   git pull
   git checkout -b feature/short-description
   ```

2. Make a focused change and run the local checks:

   ```bash
   python -m py_compile app.py web_app.py
   ```

3. Commit the change and push the branch:

   ```bash
   git add .
   git commit -m "Describe the change"
   git push -u origin feature/short-description
   ```

4. Open a Pull Request into `main`.

Direct pushes to `main` are not allowed. Every Pull Request must pass CI and receive at least one review.

## Important

Do not commit `.env` or API keys. The SQLite file is currently part of the project by design, so do not put private documents or personal data into it before publishing.

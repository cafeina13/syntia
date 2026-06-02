# Syntia — a learning Discord bot

A tiny Discord bot built to learn the basics. It has three slash commands:
`/ping`, `/hello`, and `/echo`.

## One-time setup (you do this in your browser)

You need to register the bot with Discord and get its secret **token**.

1. Go to <https://discord.com/developers/applications> and log in.
2. Click **New Application**, name it (e.g. "Syntia"), and create it.
3. In the left sidebar, click **Bot**.
4. Click **Reset Token**, then **Copy** the token. (This is your bot's password —
   keep it secret. If it ever leaks, reset it here.)
5. Copy the file `.env.example` to a new file named `.env`, and paste your token:
   ```
   DISCORD_TOKEN=your-real-token-here
   ```

## Enable the Message Content intent (needed for "syntia ..." commands)

The bot reads chat messages to detect its `syntia` prefix, which requires a
privileged intent:

1. Developer Portal -> your app -> **Bot**.
2. Scroll to **Privileged Gateway Intents**.
3. Turn ON **Message Content Intent** and save.

(Slash commands don't need this; prefix commands do.)

## Invite the bot to YOUR server

You need a Discord server you own (make one: in Discord, click the **+** on the
left, "Create My Own").

1. Back on the Developer Portal, go to **OAuth2 → URL Generator**.
2. Under **Scopes**, tick `bot` and `applications.commands`.
3. Under **Bot Permissions**, tick `Send Messages`.
4. Copy the generated URL at the bottom, open it in your browser, pick your
   server, and authorize.

## Run the bot

```powershell
.venv\Scripts\python.exe bot.py
```

You should see "Bot is ready!" in the terminal. In your Discord server, type `/`
and you'll see `ping`, `hello`, and `echo`. Try them!

> Note: slash commands can take a minute to appear the first time while Discord
> registers them. If they don't show, fully restart your Discord client.

## Project files

- `bot.py` — the bot itself (start here, it's commented line-by-line)
- `.env` — your secret token (never commit this; it's gitignored)
- `.env.example` — template showing what `.env` should look like
- `requirements.txt` — the libraries this project needs
- `.venv/` — the isolated Python 3.13 environment (gitignored)

## Reinstalling dependencies later

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

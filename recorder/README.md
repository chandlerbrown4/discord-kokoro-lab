# Server bot recorder

This is the server bot workflow. To capture through your own Discord client, follow the [desktop setup](../README.md).

1. Create a Discord bot, then install it in a server you administer with the `bot` and `applications.commands` scopes. Give the bot **View Channel** and **Connect** on a standard voice channel, and **View Channel** and **Send Messages** on the command channel. [Discord Developer Portal](https://discord.com/developers/applications)
2. Use Node.js 22.12 or later. Copy `.env.example` to `.env` and fill in the bot token and the IDs for the server and the voice and command channels. Run `npm ci` and `npm run record` from this directory.
3. A member with **Manage Server** runs `/voice-lab start` in the command channel. While in the voice channel, each participant who wants their audio captured runs `/voice-lab join` in the command channel. They can stop future capture with `/voice-lab leave` or by leaving the voice channel. The operator runs `/voice-lab stop` to finalize the session.
4. After the stop confirmation appears, pass `data/recordings/<session>` to `voice-lab prepare` from the repository root.

The bot saves 48 kHz WAV clips by Discord user ID. Each file has its own timeline. Discord.js handles DAVE voice connections and exposes a separate audio stream per opted-in user. Voice receive needs a live test in your server before a long capture. [Discord.js voice](https://discord.js.org/docs/packages/voice/0.19.2)

Review Discord's restrictions on using API data, including message content, in model training before using bot captures with Kokoro. [Discord Developer Policy](https://support-dev.discord.com/hc/en-us/articles/8563934450327-Discord-Developer-Policy) · [Discord Developer Terms](https://support-dev.discord.com/hc/en-us/articles/8562894815383-Discord-Developer-Terms-of-Service)

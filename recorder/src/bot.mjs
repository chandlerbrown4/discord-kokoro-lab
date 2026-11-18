import { resolve } from 'node:path';
import { getCiphers } from 'node:crypto';
import { ChannelType, Client, Events, GatewayIntentBits, MessageFlags, PermissionFlagsBits, SlashCommandBuilder } from 'discord.js';
import { VoiceConnectionStatus, entersState, joinVoiceChannel } from '@discordjs/voice';
import OpusScript from 'opusscript';
import { RecordingSession, SAMPLE_RATE } from './session.mjs';

const COMMAND = 'voice-lab';
const command = new SlashCommandBuilder()
  .setName(COMMAND)
  .setDescription('Manage an opt-in voice recording session.')
  .addSubcommand((sub) => sub.setName('start').setDescription('Open the recording session (Manage Server).'))
  .addSubcommand((sub) => sub.setName('join').setDescription('Agree to record your voice for a model that generates synthetic speech sounding like you.'))
  .addSubcommand((sub) => sub.setName('leave').setDescription('Stop saving your voice for this session.'))
  .addSubcommand((sub) => sub.setName('stop').setDescription('End the session and save the recordings (Manage Server).'));

function requireEnv(name, discordId = false) {
  const value = process.env[name]?.trim();
  if (!value || (discordId && !/^\d{17,20}$/.test(value))) {
    throw new Error(discordId ? `Set ${name} to a Discord ID in .env.` : `Set ${name} in .env.`);
  }
  return value;
}

function createDecoder() {
  return new OpusScript(SAMPLE_RATE, 2, OpusScript.Application.VOIP);
}

async function main() {
  const token = requireEnv('DISCORD_BOT_TOKEN');
  const guildId = requireEnv('DISCORD_GUILD_ID', true);
  const voiceChannelId = requireEnv('DISCORD_VOICE_CHANNEL_ID', true);
  const textChannelId = requireEnv('DISCORD_TEXT_CHANNEL_ID', true);
  const dataDirectory = resolve(process.env.DATA_DIR?.trim() || '../data');

  if (!getCiphers().includes('aes-256-gcm')) throw new Error('This Node build is missing AES-256-GCM.');
  createDecoder().delete();
  process.umask(0o077);

  const client = new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates],
    allowedMentions: { parse: [] },
    rest: { timeout: 10_000, retries: 1 },
  });
  let voiceChannel;
  let noticeChannel;
  let active;
  let shuttingDown = false;

  function inVoice(id) {
    return voiceChannel.guild.voiceStates.cache.get(id)?.channelId === voiceChannelId;
  }

  function end(candidate, reason) {
    if (!candidate || active !== candidate) return;
    active = undefined;
    try {
      candidate.session?.finish(reason);
    } finally {
      const conn = candidate.connection;
      if (conn && conn.state.status !== VoiceConnectionStatus.Destroyed) conn.destroy();
    }
  }

  function summary(candidate) {
    return candidate.session
      ? `Recording ended. Saved ${candidate.session.clips} audio clips locally in session \`${candidate.session.id}\`.`
      : 'Recording setup was canceled.';
  }

  async function endUnexpectedly(candidate, reason, cause) {
    if (!candidate || active !== candidate) return;
    if (cause) console.error(`Recorder ${reason}: ${cause.message}`);
    let content;
    try {
      end(candidate, reason);
      content = summary(candidate);
    } catch (error) {
      console.error(`Could not finalize recording: ${error.message}`);
      content = 'Recording ended. Check the recorder terminal for a storage error.';
    }
    if (candidate.announced) {
      try {
        await noticeChannel.send(content);
      } catch (error) {
        console.error(`Could not post stop notice: ${error.message}`);
      }
    }
  }

  async function start(interaction) {
    if (active) {
      await interaction.reply({ content: 'A session is already open or connecting.', flags: MessageFlags.Ephemeral });
      return;
    }

    const candidate = {};
    active = candidate;
    try {
      await interaction.deferReply();
      if (active !== candidate) {
        await interaction.editReply('Recording setup was canceled.');
        return;
      }
      const conn = joinVoiceChannel({
        channelId: voiceChannel.id,
        guildId,
        adapterCreator: voiceChannel.guild.voiceAdapterCreator,
        selfDeaf: false,
        selfMute: true,
      });
      candidate.connection = conn;
      conn.on('error', (error) => {
        if (candidate.session) void endUnexpectedly(candidate, 'voice_error', error);
        else console.error(`Voice setup: ${error.message}`);
      });
      conn.on('stateChange', (previous, next) => {
        if (candidate.session && previous.status === VoiceConnectionStatus.Ready &&
            next.status !== VoiceConnectionStatus.Ready) void endUnexpectedly(candidate, 'voice_disconnected');
      });
      await entersState(conn, VoiceConnectionStatus.Ready, 15_000);
      if (active !== candidate) {
        await interaction.editReply('Recording setup was canceled.');
        return;
      }
      candidate.session = new RecordingSession({
        dataDirectory,
        voiceChannelId,
        subscribe: (id) => conn.receiver.subscribe(id),
        createDecoder,
        isInChannel: inVoice,
        onFatal: (error) => void endUnexpectedly(candidate, 'storage_or_capture_error', error),
        onWarning: (id, error) => console.warn(`Dropped audio for ${id}: ${error.message}`),
      });
      await interaction.editReply(
        `Voice recording is open in <#${voiceChannelId}>. Use \`/voice-lab join\` here while in that channel to agree to save your voice locally and use it to train a model that generates synthetic speech that sounds like you. ` +
        'Use `/voice-lab leave` or leave the voice channel to stop future capture. Already saved clips remain with the session. Contact the session operator for deletion. ' +
        'You need to opt in again each time you rejoin the channel.',
      );
      if (active === candidate) candidate.announced = true;
      else await interaction.editReply(summary(candidate));
    } catch (error) {
      try {
        end(candidate, 'start_failed');
      } catch (finishError) {
        console.error(`Could not finalize failed recording: ${finishError.message}`);
      }
      throw error;
    }
  }

  async function handle(interaction) {
    if (!interaction.isChatInputCommand() || interaction.commandName !== COMMAND) return;
    if (interaction.guildId !== guildId || interaction.channelId !== textChannelId || !voiceChannel || shuttingDown) {
      await interaction.reply({ content: 'Use this command in the configured recording text channel while the recorder is ready.', flags: MessageFlags.Ephemeral });
      return;
    }

    const action = interaction.options.getSubcommand();
    const whisper = (content) => interaction.reply({ content, flags: MessageFlags.Ephemeral });
    if ((action === 'start' || action === 'stop') && !interaction.memberPermissions?.has(PermissionFlagsBits.ManageGuild)) {
      await whisper('You need Manage Server to start or stop the recording session.');
      return;
    }

    if (action === 'start') {
      await start(interaction);
      return;
    }
    if (action === 'stop') {
      const candidate = active;
      if (!candidate) {
        await whisper('There is no open recording session.');
        return;
      }
      end(candidate, 'operator_stop');
      await interaction.reply(summary(candidate));
      return;
    }

    if (!active?.announced) {
      await whisper('There is no open recording session.');
      return;
    }
    const { session } = active;
    if (action === 'join') {
      const optedIn = session.join(interaction.user.id, interaction.member?.displayName ?? interaction.user.displayName);
      await whisper(optedIn
        ? 'You opted in to recording your voice for synthetic speech that sounds like you. Use `/voice-lab leave` to stop recording.'
        : `Join <#${voiceChannelId}> first, then use \`/voice-lab join\` here.`);
      return;
    }
    const removed = session.leave(interaction.user.id);
    await whisper(removed ? 'Your voice will no longer be recorded.' : 'Your voice is not being recorded.');
  }

  client.on(Events.InteractionCreate, (interaction) => {
    void handle(interaction).catch(async (error) => {
      console.error(`Recorder command failed: ${error.message}`);
      const content = 'The recorder could not complete the command. Check the recorder terminal.';
      try {
        if (interaction.deferred) await interaction.editReply(content);
        else if (!interaction.replied) await interaction.reply({ content, flags: MessageFlags.Ephemeral });
      } catch (replyError) {
        console.error(`Could not reply: ${replyError.message}`);
      }
    });
  });

  client.on(Events.VoiceStateUpdate, (previous, next) => {
    const candidate = active;
    if (!candidate?.session || next.guild.id !== guildId || next.channelId === voiceChannelId) return;
    if (next.id === client.user.id) {
      void endUnexpectedly(candidate, 'bot_left_voice');
      return;
    }
    if (previous.channelId !== voiceChannelId) return;
    try {
      candidate.session.leave(next.id);
    } catch (error) {
      void endUnexpectedly(candidate, 'storage_error', error);
    }
  });

  async function shutdown(reason) {
    if (shuttingDown) return;
    shuttingDown = true;
    await endUnexpectedly(active, reason);
    await client.destroy();
  }

  process.once('SIGINT', () => void shutdown('sigint'));
  process.once('SIGTERM', () => void shutdown('sigterm'));
  client.on(Events.Error, (error) => {
    console.error(`Discord client: ${error.message}`);
    void endUnexpectedly(active, 'client_error');
  });
  client.on(Events.ShardDisconnect, () => void endUnexpectedly(active, 'gateway_disconnected'));

  client.once(Events.ClientReady, async () => {
    try {
      const guild = await client.guilds.fetch(guildId);
      const [voice, notice, bot] = await Promise.all([
        guild.channels.fetch(voiceChannelId),
        guild.channels.fetch(textChannelId),
        guild.members.fetchMe(),
      ]);
      if (voice?.type !== ChannelType.GuildVoice || !voice.permissionsFor(bot).has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.Connect])) {
        throw new Error('Use a server voice channel where the bot has View Channel and Connect.');
      }
      if (!notice?.isSendable() || !notice.permissionsFor(bot)?.has([PermissionFlagsBits.ViewChannel, PermissionFlagsBits.SendMessages])) {
        throw new Error('Use a server text channel where the bot has View Channel and Send Messages.');
      }
      await guild.commands.create(command);
      noticeChannel = notice;
      voiceChannel = voice;
      console.log(`Recorder ready. Use /${COMMAND} start in the configured text channel.`);
    } catch (error) {
      console.error(`Recorder setup failed: ${error.message}`);
      process.exitCode = 1;
      await shutdown('setup_failed');
    }
  });
  await client.login(token);
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});

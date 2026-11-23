// Builds the pack's workflows in the ComfyUI frontend and saves them to user/default/workflows.
// Paste into the browser console of a ComfyUI tab (or run through a devtools/automation hook): await makeWorkflows()
async function makeWorkflows() {
  const NEGATIVE = "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards";
  const PROMPT = "A western man stands on stage under dramatic lighting, holding a microphone close to their mouth. Wearing a vibrant red jacket with gold embroidery, the singer is speaking while smoke swirls around them, creating a dynamic and atmospheric scene.";
  const DMD = "longcat-video-avatar/longcat_video_avatar_1.6_dmd_lora_bf16.safetensors";
  const OPSD = "longcat-video-avatar/longcat_video_avatar_1.6_opsd_lora_bf16.safetensors";
  const FILES = `**Files**
- diffusion_models/longcat-video-avatar/longcat_video_avatar_1.6_bf16.safetensors (or _int8_convrot, _w6a8, _w4a8)
- loras/longcat-video-avatar/longcat_video_avatar_1.6_dmd_lora_bf16.safetensors and ..._opsd_lora_bf16.safetensors
- text_encoders/umt5_xxl_fp16.safetensors (Wan 2.1's), vae/wan_2.1_vae.safetensors
- audio_encoders/whisper_large_v3_encoder_fp16.safetensors

Made with tools/convert.py from [meituan-longcat/LongCat-Video-Avatar-1.6](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6).

**Size**: the release picks the closest of its 480p buckets to the image's aspect, e.g. 832x480, 768x512, 480x832.

**Audio**: give clean speech; the release separates vocals from music first.`;
  const README_ONE = `# LongCat-Video-Avatar 1.6: video, distilled

A start image, a prompt and speech in; the video out, as long as the speech. **LongCat Avatar Video
(distilled)** makes
93-frame segments (3.7 s at 25 fps), each continuing from the last 13 frames of the one before, samples them the
release's distilled way (8 steps: DMD LoRA then OPSD LoRA, euler, cfg 1) and encodes each into the mp4 as soon as it
is decoded (NVENC when there is one), so the length costs disk, not memory.

${FILES}`;
  const README_BASE = `# LongCat-Video-Avatar 1.6: video, base model

The same video as the distilled workflow, made with the base model alone (no LoRAs) the release's original way:
**LongCat Avatar Video (base)** samples each 93-frame segment with 50 steps (continued segments keep the steps
above sigma 0.5 and end on 10 even ones), text guidance 4 and audio guidance 4. Each step runs the model three times
(prompt with audio, negative with audio, negative without), so it is many times slower than the distilled workflow.

${FILES}`;

  const results = {};
  results.longcat_avatar_video = await build("longcat_avatar_video", oneNode);
  results.longcat_avatar_video_base = await build("longcat_avatar_video_base", base);
  return results;

  function common(add, link) {
    add("model", "LongCatAvatarModelLoader", { unet_name: "longcat-video-avatar/longcat_video_avatar_1.6_w4a8.safetensors" });
    add("clip", "CLIPLoader", { clip_name: "umt5_xxl_fp16.safetensors", type: "wan" });
    add("pos", "CLIPTextEncode", { text: PROMPT }, "Prompt");
    add("vae", "VAELoader", { vae_name: "wan_2.1_vae.safetensors" });
    add("image", "LoadImage", { image: "longcat_avatar_man.png" });
    add("audio", "LoadAudio", { audio: "longcat_avatar_speech_16s.wav" });
    add("whisper", "AudioEncoderLoader", { audio_encoder_name: "whisper_large_v3_encoder_fp16.safetensors" });
    add("features", "LongCatAvatarAudioEncode");
    link("clip", 0, "pos", "clip");
    link("whisper", 0, "features", "audio_encoder"); link("audio", 0, "features", "audio");
  }

  function oneNode(add, link) {
    common(add, link);
    add("video", "LongCatAvatarVideo", { dmd_lora: DMD, opsd_lora: OPSD, seed: 42, width: 768, height: 512 });
    add("save", "SaveVideo", { filename_prefix: "longcat_avatar/video" });
    link("model", 0, "video", "model"); link("pos", 0, "video", "positive"); link("vae", 0, "video", "vae");
    link("features", 0, "video", "audio_features"); link("audio", 0, "video", "audio"); link("image", 0, "video", "start_image");
    link("video", 0, "save", "video");
    return { readme: README_ONE, columns: [["readme"], ["model", "clip", "pos", "vae"], ["image", "audio", "whisper", "features"], ["video"], ["save"]] };
  }

  function base(add, link) {
    common(add, link);
    add("neg", "CLIPTextEncode", { text: NEGATIVE }, "Negative prompt");
    add("video", "LongCatAvatarVideoBase", { seed: 42, width: 768, height: 512, steps: 50, text_guidance: 4.0, audio_guidance: 4.0 });
    add("save", "SaveVideo", { filename_prefix: "longcat_avatar/video_base" });
    link("clip", 0, "neg", "clip");
    link("model", 0, "video", "model"); link("pos", 0, "video", "positive"); link("neg", 0, "video", "negative"); link("vae", 0, "video", "vae");
    link("features", 0, "video", "audio_features"); link("audio", 0, "video", "audio"); link("image", 0, "video", "start_image");
    link("video", 0, "save", "video");
    return { readme: README_BASE, columns: [["readme"], ["model", "clip", "pos", "neg", "vae"], ["image", "audio", "whisper", "features"], ["video"], ["save"]] };
  }

  async function build(name, body) {
    const graph = app.graph;
    graph.clear();
    const nodes = {};
    const add = (key, type, widgets = {}, title) => {
      const node = LiteGraph.createNode(type);
      if (!node) throw new Error(`no node type ${type}`);
      graph.add(node);
      for (const [wname, value] of Object.entries(widgets)) {
        const w = node.widgets.find((w) => w.name === wname);
        if (!w) throw new Error(`${type} has no widget ${wname}`);
        w.value = value;
      }
      if (title) node.title = title;
      nodes[key] = node;
      return node;
    };
    const link = (from, slot, to, input) => {
      const target = nodes[to];
      const index = target.inputs.findIndex((i) => i.name === input);
      if (index < 0) throw new Error(`${target.type} has no input ${input}`);
      nodes[from].connect(slot, target, index);
    };
    const note = add("readme", "MarkdownNote");
    const { readme, columns } = body(add, link);
    note.widgets[0].value = readme;
    note.size = [440, 560];
    // columns, laid out from measured sizes
    let x = 0;
    for (const column of columns) {
      let y = 0, width = 0;
      for (const key of column) {
        const node = nodes[key];
        if (key !== "readme") node.size = node.computeSize();
        if (["pos", "neg"].includes(key)) node.size = [360, 160];
        node.pos = [x, y];
        y += node.size[1] + 60;
        width = Math.max(width, node.size[0]);
      }
      x += width + 50;
    }
    graph.setDirtyCanvas(true, true);
    const res = await fetch(`/api/userdata/workflows%2F${name}.json?overwrite=true`, { method: "POST", body: JSON.stringify(graph.serialize()) });
    return { saved: res.status, nodes: Object.keys(nodes).length };
  }
}

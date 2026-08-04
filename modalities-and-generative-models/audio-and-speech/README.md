---
id: "14-audio-and-speech"
topic: "Audio & Speech"
level: advanced
built_from: ["deep-learning", "nlp"]
leads_to: ["multimodal", "agentic-ai"]
updated: 2026-07-14
---

# Audio & Speech
> Models that *listen and talk* — audio representations and neural codecs, speech recognition
> (ASR), speech synthesis (TTS), and the realtime voice stack. Audio *diffusion/music
> generation* details live with [Diffusion](../diffusion-models/README.md); this topic owns the
> speech pipeline end to end.

**⭐ Start here:** [Hugging Face Audio Course](https://huggingface.co/learn/audio-course) — **Hugging Face** — free, code-first path through representations → ASR → TTS.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page.
> **✅ ready · ⬜ coming soon.** Representations first — everything downstream stands on them.

### Representations & codecs
1. ⬜ Audio Representations (waveform · spectrogram · mel · MFCC)
2. ⬜ Audio Tokenization & Neural Codecs (SoundStream · EnCodec · discrete audio tokens)

### Speech recognition
3. ⬜ ASR Fundamentals (CTC · seq2seq · WER and its traps)
4. ⬜ Self-Supervised Speech (wav2vec 2.0 · HuBERT)
5. ⬜ Whisper & Large-Scale Weakly-Supervised ASR

### Speech synthesis
6. ⬜ TTS Fundamentals (Tacotron lineage · VITS · vocoders)
7. ⬜ Modern TTS & Voice Cloning (codec LMs · zero-shot voices · safety)

### Systems & applications
8. ⬜ Speaker Identification & Diarization
9. ⬜ Realtime & Streaming Voice (latency budgets · VAD · barge-in · voice agents)
10. ⬜ Music & Audio Generation overview (codec LMs · pointers to diffusion)

### Related concepts (covered in another section)
> Kept in their canonical home to avoid repetition.
- **Sequence decoding (beam search · CTC decoding math)** → [NLP](../natural-language-processing/README.md)
- **Audio diffusion models** → [Diffusion](../diffusion-models/README.md)
- **Audio in multimodal LLMs** → [Multimodal](../multimodal-learning/README.md)
- **Fourier analysis & signal processing math** → [Advanced Research Math · Fourier Analysis](../../specialized-studies/advanced-mathematics-for-ai-research/README.md)

## 🎓 Courses (free)
- [Hugging Face Audio Course](https://huggingface.co/learn/audio-course) — **Hugging Face** — the practical spine for this topic.

## 📄 Key Papers
- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356) — **Radford et al. (2022)** — the ASR baseline everyone deploys.
- [wav2vec 2.0](https://arxiv.org/abs/2006.11477) — **Baevski et al. (2020)** — self-supervised speech representations.
- [High Fidelity Neural Audio Compression (EnCodec)](https://arxiv.org/abs/2210.13438) — **Défossez et al. (2022)** — the codec behind audio-token LMs.

## 📚 Books
- [Speech and Language Processing — Ch. 16 (ASR & TTS)](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — free draft, the standard reference.

## 🔗 In this platform
- Language side: [NLP](../natural-language-processing/README.md) · [LLMs](../../09.%20LLMs/README.md) · Fusion: [Multimodal](../multimodal-learning/README.md)

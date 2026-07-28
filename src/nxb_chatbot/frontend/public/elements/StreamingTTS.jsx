import { useRef, useState } from "react";

export default function StreamingTTS() {
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const audioContextRef = useRef(null);
  const socketRef = useRef(null);
  const nextStartTimeRef = useRef(0);
  const activeSourcesRef = useRef([]);

  const playPcmChunk = (
    arrayBuffer,
    sampleRate,
    channels
  ) => {
    const audioContext = audioContextRef.current;

    if (!audioContext) {
      return;
    }

    const pcm = new Int16Array(arrayBuffer);

    const frameCount = Math.floor(
      pcm.length / channels
    );

    if (frameCount === 0) {
      return;
    }

    const audioBuffer = audioContext.createBuffer(
      channels,
      frameCount,
      sampleRate
    );

    for (
      let channelIndex = 0;
      channelIndex < channels;
      channelIndex += 1
    ) {
      const output =
        audioBuffer.getChannelData(channelIndex);

      for (
        let frameIndex = 0;
        frameIndex < frameCount;
        frameIndex += 1
      ) {
        const pcmIndex =
          frameIndex * channels + channelIndex;

        output[frameIndex] =
          pcm[pcmIndex] / 32768;
      }
    }

    const source = audioContext.createBufferSource();

    source.buffer = audioBuffer;
    source.connect(audioContext.destination);

    const minimumStartTime =
      audioContext.currentTime + 0.05;

    if (
      nextStartTimeRef.current <
      minimumStartTime
    ) {
      nextStartTimeRef.current =
        minimumStartTime;
    }

    source.start(nextStartTimeRef.current);

    nextStartTimeRef.current +=
      audioBuffer.duration;

    activeSourcesRef.current.push(source);

    source.onended = () => {
      activeSourcesRef.current =
        activeSourcesRef.current.filter(
          (activeSource) =>
            activeSource !== source
        );
    };
  };

  const stopAudio = async () => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    activeSourcesRef.current.forEach(
      (source) => {
        try {
          source.stop();
        } catch {
          // Source may already be stopped.
        }
      }
    );

    activeSourcesRef.current = [];

    if (audioContextRef.current) {
      await audioContextRef.current.close();
      audioContextRef.current = null;
    }

    nextStartTimeRef.current = 0;
    setStatus("idle");
  };

  const startAudio = async () => {
    if (
      status === "connecting" ||
      status === "playing"
    ) {
      return;
    }

    setError("");
    setStatus("connecting");

    const AudioContextClass =
      window.AudioContext ||
      window.webkitAudioContext;

    if (!AudioContextClass) {
      setError(
        "This browser does not support Web Audio."
      );
      setStatus("error");
      return;
    }

    const audioContext =
      new AudioContextClass();

    audioContextRef.current = audioContext;

    await audioContext.resume();

    nextStartTimeRef.current =
      audioContext.currentTime + 0.1;

    const socket = new WebSocket(
      props.websocketUrl
    );

    socket.binaryType = "arraybuffer";
    socketRef.current = socket;

    let sampleRate = 22050;
    let channels = 1;

    socket.onopen = () => {
      socket.send(
        JSON.stringify({
          text: props.text,
        })
      );
    };

    socket.onmessage = (event) => {
      if (typeof event.data === "string") {
        const message = JSON.parse(event.data);

        if (message.type === "started") {
          setStatus("connecting");
        }

        if (message.type === "format") {
          sampleRate =
            message.sample_rate || 22050;

          channels =
            message.channels || 1;

          setStatus("playing");
        }

        if (message.type === "done") {
          setStatus("finished");
          socket.close();
        }

        if (message.type === "error") {
          setError(
            message.message ||
              "TTS streaming failed."
          );

          setStatus("error");
          socket.close();
        }

        return;
      }

      playPcmChunk(
        event.data,
        sampleRate,
        channels
      );
    };

    socket.onerror = () => {
      setError(
        "Could not connect to the TTS service."
      );
      setStatus("error");
    };

    socket.onclose = () => {
      socketRef.current = null;
    };
  };

  const handleClick = async () => {
    if (
      status === "connecting" ||
      status === "playing"
    ) {
      await stopAudio();
      return;
    }

    await startAudio();
  };

  const getButtonText = () => {
    if (status === "connecting") {
      return "Preparing speech...";
    }

    if (status === "playing") {
      return "Stop audio";
    }

    if (status === "finished") {
      return "Play again";
    }

    if (status === "error") {
      return "Retry audio";
    }

    return "Listen to response";
  };

  return (
    <div className="flex flex-col gap-2 py-2">
      <button
        onClick={handleClick}
        className="
          w-fit rounded-md border
          px-3 py-2 text-sm
          hover:bg-muted
        "
      >
        🔊 {getButtonText()}
      </button>

      {status === "playing" && (
        <span className="text-xs text-muted-foreground">
          Audio is streaming...
        </span>
      )}

      {error && (
        <span className="text-xs text-red-500">
          {error}
        </span>
      )}
    </div>
  );
}
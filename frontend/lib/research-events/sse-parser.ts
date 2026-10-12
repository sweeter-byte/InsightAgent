export type ParsedSseFrame = {
  id: string;
  event: string;
  data: string;
};

type FrameState = {
  event: string;
  dataLines: string[];
  hasData: boolean;
};

function emptyFrame(): FrameState {
  return {
    event: "message",
    dataLines: [],
    hasData: false,
  };
}

export function createSseParser(
  onEvent: (event: ParsedSseFrame) => void,
): { feed(chunk: string): void; finish(): void } {
  let lineBuffer = "";
  let frame = emptyFrame();
  let lastEventId = "";
  let finished = false;

  function dispatchFrame() {
    const eventToDispatch = frame.hasData
      ? {
          id: lastEventId,
          event: frame.event || "message",
          data: frame.dataLines.join("\n"),
        }
      : null;

    frame = emptyFrame();

    if (eventToDispatch) {
      onEvent(eventToDispatch);
    }
  }

  function processLine(line: string) {
    if (line === "") {
      dispatchFrame();
      return;
    }

    if (line.startsWith(":")) {
      return;
    }

    const colonIndex = line.indexOf(":");
    const field = colonIndex === -1 ? line : line.slice(0, colonIndex);
    let value = colonIndex === -1 ? "" : line.slice(colonIndex + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }

    switch (field) {
      case "data":
        frame.dataLines.push(value);
        frame.hasData = true;
        break;
      case "event":
        frame.event = value;
        break;
      case "id":
        if (!value.includes("\0")) {
          lastEventId = value;
        }
        break;
    }
  }

  return {
    feed(chunk) {
      if (finished) {
        return;
      }

      lineBuffer += chunk;
      let newlineIndex = lineBuffer.indexOf("\n");

      while (newlineIndex !== -1) {
        let line = lineBuffer.slice(0, newlineIndex);
        lineBuffer = lineBuffer.slice(newlineIndex + 1);

        if (line.endsWith("\r")) {
          line = line.slice(0, -1);
        }

        processLine(line);
        newlineIndex = lineBuffer.indexOf("\n");
      }
    },
    finish() {
      lineBuffer = "";
      frame = emptyFrame();
      finished = true;
    },
  };
}

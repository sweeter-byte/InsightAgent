import "@testing-library/jest-dom/vitest";

import { afterAll, afterEach, beforeAll } from "vitest";

import { mockServer } from "./tests/mocks/server";

beforeAll(() => mockServer.listen({ onUnhandledFrame: "error" }));
afterEach(() => mockServer.resetHandlers());
afterAll(() => mockServer.close());

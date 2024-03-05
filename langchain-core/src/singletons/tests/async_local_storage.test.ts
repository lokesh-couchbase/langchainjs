import { test, expect } from "@jest/globals";
import { AsyncLocalStorage } from "node:async_hooks";
import { AsyncLocalStorageProviderSingleton } from "../index.js";
import { RunnableLambda } from "../../runnables/base.js";

test("Config should be automatically populated after setting global async local storage", async () => {
  const inner = RunnableLambda.from((_, config) => config);
  const outer = RunnableLambda.from(async (input) => {
    const res = await inner.invoke(input);
    return res;
  });
  const res1 = await outer.invoke(
    { hi: true },
    {
      configurable: {
        sampleKey: "sampleValue",
      },
      tags: ["tester"],
    }
  );
  expect(res1?.tags).toEqual([]);
  AsyncLocalStorageProviderSingleton.initializeGlobalInstance(
    new AsyncLocalStorage()
  );
  const res2 = await outer.invoke(
    { hi: true },
    {
      configurable: {
        sampleKey: "sampleValue",
      },
      tags: ["tester"],
    }
  );
  expect(res2?.tags).toEqual(["tester"]);

  const stream = await outer.stream(
    { hi2: true },
    {
      configurable: {
        sampleKey: "sampleValue",
      },
      tags: ["stream_tester"],
    }
  );
  const chunks = [];
  for await (const chunk of stream) {
    console.log(chunk);
    chunks.push(chunk);
  }
  expect(chunks.length).toEqual(1);
  expect(chunks[0]).toEqual(
    expect.objectContaining({
      configurable: {
        sampleKey: "sampleValue",
      },
      tags: ["stream_tester"],
    })
  );

  const outer2 = RunnableLambda.from(async () => inner);
  const res3 = await outer2.invoke(
    {},
    {
      configurable: {
        sampleKey: "sampleValue",
      },
      tags: ["test_recursive"],
    }
  );
  expect(res3?.tags).toEqual(["test_recursive"]);
  const stream2 = await outer2.stream(
    {},
    {
      configurable: {
        sampleKey: "sampleValue",
      },
      tags: ["stream_test_recursive"],
    }
  );
  const chunks2 = [];
  for await (const chunk of stream2) {
    console.log(chunk);
    chunks2.push(chunk);
  }
  expect(chunks2.length).toEqual(1);
  expect(chunks2[0]).toEqual(
    expect.objectContaining({
      configurable: {
        sampleKey: "sampleValue",
      },
      tags: ["stream_test_recursive"],
    })
  );

  const inner2 = RunnableLambda.from((_, config) => config).withConfig({
    runName: "inner_test_run",
  });
  const outer3 = RunnableLambda.from(async (input) => {
    const res = await inner2.invoke(input);
    return res;
  });

  const res4 = await outer3.invoke(
    { hi: true },
    {
      configurable: {
        sampleKey: "sampleValue",
      },
      tags: ["tester_with_config"],
    }
  );
  expect(res4?.tags).toEqual(["tester_with_config"]);
});

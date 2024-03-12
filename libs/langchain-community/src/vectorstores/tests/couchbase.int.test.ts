/* eslint-disable no-process-env */
import { describe, test } from "@jest/globals";
import { Cluster } from "couchbase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Document } from "@langchain/core/documents";
import {
  CouchbaseVectorStore,
  CouchbaseVectorStoreArgs,
} from "../couchbase.js";

describe("Couchbase vector store", () => {
  const connectionString = process.env.DB_CONN_STR || "localhost";
  const databaseUsername = process.env.DB_USERNAME;
  const databasePassword = process.env.DB_PASSWORD;
  const bucketName = "movies-clone";
  const scopeName = "test2";
  const collectionName = "col1";
  const indexName = "movies-clone-test";
  const textFieldKey = "text";
  const embeddingFieldKey = "embedding";
  const isScopedIndex = true;
  let couchbaseClient: Cluster;

  const texts = [
    "Couchbase, built on a key-value store, offers efficient data operations.",
    "As a NoSQL database, Couchbase provides scalability and flexibility to handle diverse data types.",
    "Couchbase supports N1QL, a SQL-like language, easing the transition for developers familiar with SQL.",
    "Couchbase ensures high availability with built-in fault tolerance and automatic multi-master replication.",
    "With its memory-first architecture, Couchbase delivers high performance and low latency data access.",
  ];

  const metadata = [
    { id: "101", name: "Efficient Operator" },
    { id: "102", name: "Flexible Storer" },
    { id: "103", name: "Quick Performer" },
    { id: "104", name: "Reliable Guardian" },
    { id: "105", name: "Adaptable Navigator" },
  ];

  beforeEach(async () => {
    couchbaseClient = await Cluster.connect(connectionString, {
      username: databaseUsername,
      password: databasePassword,
      configProfile: "wanDevelopment",
    });
  });

  test("from Texts to vector store", async () => {
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const couchbaseConfig: CouchbaseVectorStoreArgs = {
      cluster: couchbaseClient,
      bucketName,
      scopeName,
      collectionName,
      indexName,
      textKey: textFieldKey,
      embeddingKey: embeddingFieldKey,
      scopedIndex: isScopedIndex,
    };

    const store = await CouchbaseVectorStore.fromTexts(
      texts,
      metadata,
      embeddings,
      couchbaseConfig
    );
    const results = await store.similaritySearchWithScore(texts[0], 1);

    expect(results.length).toEqual(1);
    expect(results[0][0].pageContent).toEqual(texts[0]);
    expect(results[0][0].metadata.name).toEqual(metadata[0].name);
    expect(results[0][0].metadata.id).toEqual(metadata[0].id);
  });

  test.skip("Add and delete Documents to vector store", async () => {
    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const couchbaseConfig: CouchbaseVectorStoreArgs = {
      cluster: couchbaseClient,
      bucketName,
      scopeName,
      collectionName,
      indexName,
      textKey: textFieldKey,
      embeddingKey: embeddingFieldKey,
      scopedIndex: isScopedIndex,
    };

    const documents: Document[] = [];
    for (let i = 0; i < texts.length; i += 1) {
      documents.push({
        pageContent: texts[i],
        metadata: {},
      });
    }

    const store = await CouchbaseVectorStore.initialize(
      embeddings,
      couchbaseConfig
    );
    const ids = await store.addDocuments(documents, {
      ids: metadata.map((val) => val.id),
      metadata: metadata.map((val) => {
        const metadataObj = {
          name: val.name,
        };
        return metadataObj;
      }),
    });

    expect(ids.length).toEqual(texts.length);
    for (let i = 0; i < ids.length; i += 1) {
      expect(ids[i]).toEqual(metadata[i].id);
    }

    const results = await store.similaritySearch(texts[1], 1);

    expect(results.length).toEqual(1);
    expect(results[0].pageContent).toEqual(texts[1]);
    expect(results[0].metadata.name).toEqual(metadata[1].name);

    await store.delete(ids);
    const cbCollection = couchbaseClient.bucket(bucketName).scope(scopeName).collection(collectionName)
    expect((await cbCollection.exists(ids[0])).exists).toBe(false)
    expect((await cbCollection.exists(ids[4])).exists).toBe(false)

    const resultsDeleted = await store.similaritySearch(texts[1], 1);
    expect(resultsDeleted.length).not.toEqual(1);
  });
});

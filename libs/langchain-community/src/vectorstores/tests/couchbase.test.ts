/* eslint-disable no-process-env */
import { test } from "@jest/globals";
import { Cluster } from "couchbase";
import { OpenAIEmbeddings } from "@langchain/openai";
import { faker } from "@faker-js/faker";

import {
  CouchbaseVectorStore,
  CouchbaseVectorStoreArgs,
} from "../couchbase.js";




test("Test Couchbase Cluster connection ", async () => {
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

  const couchbaseClient = await Cluster.connect(connectionString, {
    username: databaseUsername,
    password: databasePassword,
    configProfile: "wanDevelopment",
  });

  console.log("connected");

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

  const couchbaseVectorStore = await CouchbaseVectorStore.initialize(embeddings, couchbaseConfig);
  
  const pageContent = faker.lorem.sentence(5);
  console.log(pageContent);
  // // await couchbaseVectorStore.addDocuments([{ pageContent, metadata: { foo: "bar" } }])
  // // await CouchbaseVectorStore.fromDocuments(docs,embeddings, couchbaseConfig)
  const docsWithScore = await couchbaseVectorStore.similaritySearchWithScore("titanic is bad for climate",4)
  console.log( docsWithScore); 
  // expect(docsWithScore.length).toBeGreaterThan(0);
});

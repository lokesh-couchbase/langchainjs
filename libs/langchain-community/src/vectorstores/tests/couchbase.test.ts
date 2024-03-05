/* eslint-disable no-process-env */
import { expect, test } from "@jest/globals";
import { Cluster } from "couchbase";
import { OpenAIEmbeddings } from "@langchain/openai";
// import { faker } from "@faker-js/faker";
import { CouchbaseVectorSearch } from "../couchbase.js";

test("Test Couchbase Cluster connection ", async () => {
  const connectionString = process.env.DB_CONN_STR || "localhost";
  const databaseUsername = process.env.DB_USERNAME; 
  const databasePassword = process.env.DB_PASSWORD;

  const couchbaseClient = await Cluster.connect(connectionString, {
    username: databaseUsername,
    password: databasePassword,
    configProfile: "wanDevelopment",
  });

  console.log("connected");

  const embeddings = new OpenAIEmbeddings({openAIApiKey: process.env.OPENAI_API_KEY}) 
  const couchbaseVectorStore = new CouchbaseVectorSearch(couchbaseClient,"movies-clone","testing", "1024",embeddings,"movies-clone","overview", "overview_embedding")
  // const pageContent = faker.lorem.sentence(5);
  // await couchbaseVectorStore.addDocuments([{ pageContent, metadata: { foo: "bar" } }])
  const docsWithScore = await couchbaseVectorStore.similaritySearch("Dinosaurs are being artificially created in a park where");
  console.log(docsWithScore)
  expect(docsWithScore.length).toBeGreaterThan(0);
});

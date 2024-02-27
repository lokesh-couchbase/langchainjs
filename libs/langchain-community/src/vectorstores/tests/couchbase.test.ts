import { expect, test } from "@jest/globals";
import { Cluster } from "couchbase";
import { OpenAIEmbeddings } from "@langchain/openai";
// import { faker } from "@faker-js/faker";
import { CouchbaseVectorSearch } from "../couchbase.js";

test("Test Couchbase Cluster connection ", async () => {
  const connectionString = "couchbase://3.76.104.168";
  const databaseUsername = "Administrator"; 
  const databasePassword = "P@ssword1!";
  // const query = `
  //   SELECT h.* FROM \`travel-sample\`.inventory.hotel h 
  //   WHERE h.country = 'United States'
  //   LIMIT 10
  // `;
  // const validPageContentFields = ["country", "name", "description"];
  // const validMetadataFields = ["id"];

  const couchbaseClient = await Cluster.connect(connectionString, {
    username: databaseUsername,
    password: databasePassword,
    configProfile: "wanDevelopment",
  });

  console.log("connected");

  const embeddings = new OpenAIEmbeddings({openAIApiKey: "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"});
  const couchbaseVectorStore = new CouchbaseVectorSearch(couchbaseClient,"movies-clone","testing", "1024",embeddings,"movies-clone","overview", "overview-embeddings")
  // const pageContent = faker.lorem.sentence(5);
  // await couchbaseVectorStore.addDocuments([{ pageContent, metadata: { foo: "bar" } }])
  const docsWithScore = await couchbaseVectorStore.similaritySearch("Star Wars");
  expect(docsWithScore.length).toBeGreaterThan(0);
});

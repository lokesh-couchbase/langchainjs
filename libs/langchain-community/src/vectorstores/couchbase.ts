/* eslint-disable @typescript-eslint/no-explicit-any */
import { Embeddings } from "@langchain/core/embeddings";
import { VectorStore } from "@langchain/core/vectorstores";
import {
  Bucket,
  Cluster,
  Collection,
  Scope,
  SearchRequest,
  VectorQuery,
  VectorSearch,
} from "couchbase";
import { Document, DocumentInterface } from "@langchain/core/documents";
import { v4 as uuid } from "uuid";

export interface AddVectorOptions {
  ids?: string[];
  metadata?: string[];
}

type CouchbaseVectorStoreFilter = { [key: string]: any };

export class CouchbaseVectorSearch extends VectorStore {
  declare FilterType: CouchbaseVectorStoreFilter;

  private readonly cluster: Cluster;

  private readonly _bucket: Bucket;

  private readonly _scope: Scope;

  private readonly _collection: Collection;

  private readonly bucketName: string;

  private readonly scopeName: string;

  private readonly collectionName: string;

  private readonly indexName: string;

  private readonly textKey: string;

  private readonly embeddingKey: string;

  private readonly scopedIndex: boolean;

  private readonly metadataKey = "metadata";

  constructor(
    cluster: Cluster,
    bucketName: string,
    scopeName: string,
    collectionName: string,
    embedding: Embeddings,
    indexName: string,
    textKey = "text",
    embeddingKey: string | undefined = undefined,
    scopedIndex = true
  ) {
    super(embedding, embedding);
    this.cluster = cluster;
    this.bucketName = bucketName;
    this.scopeName = scopeName;
    this.collectionName = collectionName;
    this.indexName = indexName;
    this.textKey = textKey;
    if (embeddingKey) {
      this.embeddingKey = embeddingKey;
    } else {
      this.embeddingKey = `${textKey}_embedding`;
    }
    this.scopedIndex = scopedIndex;

    this._bucket = this.cluster.bucket(this.bucketName);
    this._scope = this._bucket.scope(this.scopeName);
    this._collection = this._scope.collection(this.collectionName);

    void this.verifyIndexes();
  }

  async verifyIndexes() {
    if (this.scopedIndex) {
      const allIndexes = await this._scope.searchIndexes().getAllIndexes();
      const indexNames = allIndexes.map((index) => index.name);
      if (!indexNames.includes(this.indexName)) {
        throw new Error(
          `Index ${this.indexName} does not exist. Please create the index before searching.`
        );
      }
    } else {
      const allIndexes = await this.cluster.searchIndexes().getAllIndexes();
      const indexNames = allIndexes.map((index) => index.name);
      if (!indexNames.includes(this.indexName)) {
        throw new Error(
          `Index ${this.indexName} does not exist. Please create the index before searching.`
        );
      }
    }
  }

  _vectorstoreType(): string {
    return "couchbase";
  }

  public async addVectors(
    vectors: number[][],
    documents: Document[],
    options: AddVectorOptions = {}
  ): Promise<string[]> {
    // Get document ids. if ids are not available then use UUIDs for each document
    let ids: string[] | undefined = options ? options.ids : undefined;
    if (ids === undefined) {
      ids = Array.from({ length: documents.length }, () => uuid());
    }

    // Get metadata for each document. if metadata is not available, use empty object for each document
    let metadata: any = options ? options.metadata : undefined;
    if (metadata === undefined) {
      metadata = Array.from({ length: documents.length }, () => ({}));
    }

    const documentsToInsert = ids.map((id: string, index: number) => ({
      [id]: {
        [this.textKey]: documents[index],
        [this.embeddingKey]: vectors[index],
        [this.metadataKey]: metadata[index],
      },
    }));

    const docIds: string[] = [];
    for (const document of documentsToInsert) {
      try {
        const currentDocumentKey = Object.keys(document)[0];
        await this._collection.upsert(
          currentDocumentKey,
          document[currentDocumentKey]
        );
        docIds.push(currentDocumentKey);
      } catch (e) {
        console.log("error received while upserting document", e);
      }
    }

    return docIds;
  }

  async similaritySearchVectorWithScore(
    embeddings: number[],
    k = 4,
    filter: CouchbaseVectorStoreFilter = {},
    fetchK = 20,
    kwargs: { [key: string]: any } = {}
  ): Promise<[DocumentInterface<Record<string, any>>, number][]> {
    let { fields } = kwargs;
    if (!fields) {
      fields = [this.textKey, this.metadataKey];
    }

    const searchRequest = new SearchRequest(
      VectorSearch.fromVectorQuery(
        new VectorQuery(this.embeddingKey, embeddings).numCandidates(fetchK)
      )
    );

    let searchIterator;
    const docsWithScore: [DocumentInterface<Record<string, any>>, number][] =
      [];

    try {
      if (this.scopedIndex) {
        searchIterator = this._scope.search(this.indexName, searchRequest, {
          limit: k,
          fields: [this.textKey, "metadata"],
          raw: filter,
        });
      } else {
        searchIterator = this.cluster.search(this.indexName, searchRequest, {
          limit: k,
          fields: [this.textKey, "metadata"],
          raw: filter,
        });
      }

      const searchRows = (await searchIterator).rows;
      for (const row of searchRows) {
        const text = row.fields[this.textKey];
        delete row.fields[this.textKey];
        const metadataField = row.fields;
        const searchScore = row.score;
        const doc = new Document({
          pageContent: text,
          metadata: metadataField,
        });
        docsWithScore.push([doc, searchScore]);
      }
    } catch (err) {
      throw new Error(`Search failed with error: ${err}`);
    }
    return docsWithScore;
  }

  async similaritySearchByVector(
    embeddings: number[],
    k = 4,
    filter: CouchbaseVectorStoreFilter = {},
    fetchK = 20,
    kwargs: { [key: string]: any } = {}
  ): Promise<Document[]> {
    const docsWithScore = await this.similaritySearchVectorWithScore(
      embeddings,
      k,
      filter,
      fetchK,
      kwargs
    );
    const docs = [];
    for (const doc of docsWithScore) {
      docs.push(doc[0]);
    }
    return docs;
  }

  async similaritySearch(
    query: string,
    k = 4,
    filter: CouchbaseVectorStoreFilter = {}
  ): Promise<Document[]> {
    const docsWithScore = await this.similaritySearchWithScore(
      query,
      k,
      filter
    );
    const docs = [];
    for (const doc of docsWithScore) {
      docs.push(doc[0]);
    }
    return docs;
  }

  async similaritySearchWithScore(
    query: string,
    k = 4,
    filter: CouchbaseVectorStoreFilter = {}
  ): Promise<[DocumentInterface<Record<string, any>>, number][]> {
    const embeddings = await this.embeddings.embedQuery(query);
    const docsWithScore = await this.similaritySearchVectorWithScore(
      embeddings,
      k,
      filter
    );
    return docsWithScore;
  }

  public async addDocuments(
    documents: Document[],
    options: AddVectorOptions = {}
  ) {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      options
    );
  }
}

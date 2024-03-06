/* eslint-disable no-param-reassign */
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

/**
 * This interface define the optional fields for adding vector
 */
export interface AddVectorOptions {
  ids?: string[];
  metadata?: Record<string, any>[];
}

type CouchbaseVectorStoreFilter = { [key: string]: any };

/**
 * Class for interacting with the Couchbase database. It extends the
 * VectorStore class and provides methods for adding vectors and
 * documents, and searching for similar vectors
 */
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

  /**
   * Class for interacting with the Couchbase database.
   * It extends the VectorStore class and provides methods
   * for adding vectors and documents, and searching for similar vectors.
   * This also verifies the index
   *
   * @param cluster - The Couchbase cluster that the store will interact with.
   * @param bucketName - The name of the bucket in the Couchbase cluster.
   * @param scopeName - The name of the scope within the bucket.
   * @param collectionName - The name of the collection within the scope.
   * @param embedding - The embeddings to be used for vector operations.
   * @param indexName - The name of the index to be used for vector search.
   * @param textKey - The key to be used for text in the documents. Defaults to "text".
   * @param embeddingKey - The key to be used for embeddings in the documents. If not provided, defaults to undefined.
   * @param scopedIndex - Whether to use a scoped index for vector search. Defaults to true.
   */
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

  /**
   * An asynchrononous method to verify the search indexes.
   * It retrieves all indexes and checks if specified index is present.
   *
   * @throws {Error} If the specified index does not exist in the database.
   */
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

  /**
   * Add vectors and corresponding documents to a couchbase collection
   * If the document IDs are passed, the existing documents (if any) will be
   * overwritten with the new ones.
   * @param vectors - The vectors to be added to the collection.
   * @param documents - The corresponding documents to be added to the collection.
   * @param options - Optional parameters for adding vectors.
   * This may include the IDs and metadata of the documents to be added. Defaults to an empty object.
   *
   * @returns - A promise that resolves to an array of document IDs that were added to the collection.
   */
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

  /**
   * Performs a similarity search on the vectors in the Couchbase database and returns the documents and their corresponding scores.
   *
   * @param embeddings - Embedding vector to look up documents similar to.
   * @param k - Number of documents to return. Defaults to 4.
   * @param filter - Optional search filter that are passed to Couchbase search. Defaults to empty object
   * @param kwargs - Optional list of fields to include in the
   * metadata of results. Note that these need to be stored in the index.
   * If nothing is specified, defaults to document metadata fields.
   *
   * @returns - Promise of list of [document, score] that are the most similar to the query vector.
   *
   * @throws If the search operation fails.
   */
  async similaritySearchVectorWithScore(
    embeddings: number[],
    k = 4,
    filter: CouchbaseVectorStoreFilter = {},
    kwargs: { [key: string]: any } = {}
  ): Promise<[DocumentInterface<Record<string, any>>, number][]> {
    let { fields } = kwargs;

    if (!fields) {
      fields = [this.textKey, this.metadataKey];
    }

    // Document text field needs to be returned from the search
    if (!fields.include(this.textKey)) {
      fields.push(this.textKey);
    }

    const searchRequest = new SearchRequest(
      VectorSearch.fromVectorQuery(
        new VectorQuery(this.embeddingKey, embeddings).numCandidates(k)
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

  /**
   * Return documents that are most similar to the vector embedding.
   *
   * @param embeddings - Embedding to look up documents similar to.
   * @param k - The number of similar documents to return. Defaults to 4.
   * @param filter - Optional search options that are passed to Couchbase search. Defaults to empty object.
   * @param kwargs - Optional list of fields to include in the metadata of results.
   * Note that these need to be stored in the index.
   * If nothing is specified, defaults to document text and metadata fields.
   *
   * @returns - A promise that resolves to an array of documents that match the similarity search.
   */
  async similaritySearchByVector(
    embeddings: number[],
    k = 4,
    filter: CouchbaseVectorStoreFilter = {},
    kwargs: { [key: string]: any } = {}
  ): Promise<Document[]> {
    const docsWithScore = await this.similaritySearchVectorWithScore(
      embeddings,
      k,
      filter,
      kwargs
    );
    const docs = [];
    for (const doc of docsWithScore) {
      docs.push(doc[0]);
    }
    return docs;
  }

  /**
   * Return documents that are most similar to the query.
   *
   * @param query - Query to look up for similar documents
   * @param k - The number of similar documents to return. Defaults to 4.
   * @param filter - Optional search options that are passed to Couchbase search. Defaults to empty object.
   *
   * @returns - Promise of list of documents that are most similar to the query.
   */
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

  /**
   * Return documents that are most similar to the query with their scores.
   *
   * @param query - Query to look up for similar documents
   * @param k - The number of similar documents to return. Defaults to 4.
   * @param filter - Optional search options that are passed to Couchbase search. Defaults to empty object.
   *
   * @returns - Promise of list of documents that are most similar to the query.
   */
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

  /**
   * Run texts through the embeddings and persist in vectorstore.
   * If the document IDs are passed, the existing documents (if any) will be
   * overwritten with the new ones.
   * @param documents - The corresponding documents to be added to the collection.
   * @param options - Optional parameters for adding documents.
   * This may include the IDs and metadata of the documents to be added. Defaults to an empty object.
   *
   * @returns - A promise that resolves to an array of document IDs that were added to the collection.
   */
  public async addDocuments(
    documents: Document[],
    options: AddVectorOptions = {}
  ) {
    const texts = documents.map(({ pageContent }) => pageContent);
    const metadatas = documents.map((doc) => doc.metadata);
    if (!options.metadata) {
      options.metadata = metadatas;
    }
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      options
    );
  }
}

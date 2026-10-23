package org.chat.supportpilot;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.document.Document;

import java.util.List;

@SpringBootTest
class VectorStoreTest {

    @Autowired
    private VectorStore vec;

    @Test
    void testFaqRetrieval() {
        List<Document> docs = vec.similaritySearch(
                SearchRequest.builder()
                        .query("What is the refund policy?")
                        .topK(4)
                        .similarityThreshold(0.0)
                        .build()
        );

        System.out.println("Direct search result count = " + docs.size());

        docs.forEach(doc ->
                System.out.println(
                        "Content: " + doc.getText()
                                + "\nMetadata: " + doc.getMetadata()
                )
        );
    }
}

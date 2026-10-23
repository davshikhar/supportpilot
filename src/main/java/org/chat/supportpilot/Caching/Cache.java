package org.chat.supportpilot.Caching;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import redis.clients.jedis.RedisClient;

import org.springframework.ai.document.Document;

import java.util.*;
import java.util.regex.Pattern;

@Component
public class Cache {
    public record Hit(String answer, List<String> sources, double score){}

    private static final Logger log = LoggerFactory.getLogger(Cache.class);
    private static final Pattern ORDER_REF = Pattern.compile("(?i)\\bORD-\\d+");

    private final VectorStore vectorStore;
    private final RedisClient redisClient;
    private final double threshold;
    private final long ttlSeconds;
    public Cache(@Qualifier("cacheVectorStore") VectorStore vec,
                 RedisClient client,
                 @Value("${supportpilot.cache.threshold:0.85}") double threshold,
                 @Value("${supportpilot.cache.ttl-minutes:60}") long ttl){
        this.vectorStore = vec;this.redisClient = client;
        this.threshold = threshold; this.ttlSeconds = ttl*60;
    }

    public boolean isCacheable(String ques){
        return !ORDER_REF.matcher(ques).find();
    }

    public Optional<Hit> look(String ques){
        try{
            List<Document> found = vectorStore.similaritySearch(SearchRequest.builder().query(ques).topK(1).build());
            if(found.isEmpty()){
                log.info("cache Miss: cache is Empty");
                return Optional.empty();
            }
            System.out.println(found);
            Document bestFit = found.get(0);
            double score = bestFit.getScore() == null ? 0 : bestFit.getScore();
            boolean hit = score >= threshold;
            log.info("cache {}: bestScore={} threshold={} matched='{}'",
                    hit? "HIT":"MISS",score,threshold,bestFit.getText());
            if(!hit){
                return Optional.empty();
            }
            Object answer = bestFit.getMetadata().get("answer");
            Object sources = bestFit.getMetadata().get("sources");
            if(!(answer instanceof String a) || a.isBlank()){
                return Optional.empty();
            }
            List<String> sourcesList = (sources instanceof String s && !s.isBlank())
                    ?List.of(s.split(",")):List.of();
            return Optional.of(new Hit(a,sourcesList,score));
        }
        catch(Exception e){
            log.warn("cache lookup failed, going ahead without cache: {}",e.getMessage());
            return Optional.empty();
        }
    }
    public void store(String ques, String ans, List<String> sources){
        log.info(">>> STORING CACHE ENTRY for question: {}", ques);
        try{
            String id = UUID.randomUUID().toString();
            Document d = new Document(id,ques, Map.<String, Object>of("answer", ans, "sources", String.join(",", sources)));
            log.info("Storing cache entry for question: {}", ques);
            vectorStore.add(List.of(d));
            redisClient.expire("cache:"+id, ttlSeconds);
        }
        catch(Exception e){
            log.warn("cache store failed: {}", e.getMessage());
        }
    }

    public void clear(){
        Set<String> keys = redisClient.keys("cache:*");
        if(!keys.isEmpty()){
            redisClient.del(keys.toArray(new String[0]));
        }
    }

}

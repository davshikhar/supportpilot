package org.chat.supportpilot.Controller;

import io.micrometer.core.instrument.*;
import jakarta.validation.Valid;
import org.chat.supportpilot.Auth;
import org.chat.supportpilot.Caching.Cache;
import org.chat.supportpilot.Records.ChatReply;
import org.chat.supportpilot.Records.ChatRequest;
import org.chat.supportpilot.ToolRouter;
import org.chat.supportpilot.Tools.OrderTool;
import org.chat.supportpilot.Tools.RefundTool;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.ChatClientResponse;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.MessageType;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.document.Document;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.http.MediaType;
import org.springframework.http.codec.ServerSentEvent;
import org.springframework.web.bind.annotation.*;
import reactor.core.publisher.Flux;

import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

@RestController
@RequestMapping("/api")
public class ChatController {

    private static final String p="""
        You are SupportPilot, a customer support assistant. Keep answers short and polite.
        Answer general questions only from the retrieved document context. Never invent policy details.
        Match questions to the context by meaning, not exact wording. For example, if the context says "refunds go back to the original payment method within 5 to 7 days", then "How long do refunds take?" or "What is the refund processing time?" should be answered "Refunds go back to your original payment method within 5 to 7 days after we receive the item."
        If the context contains the answer, state it directly and do not mention a human agent. Only if the context truly does not contain the answer, say you don't have that information and offer to connect the customer with a human agent. Never say you don't have the information and then answer anyway.
        If a tool is available for the request, use it: getOrderStatus for an order's status, requestRefund for a refund.
        If the customer asks about an order's status or wants a refund and you have no tool for it, ask them to send the full request in one message, for example "What is the status of order ORD-1234?" or "I want a refund for ORD-1234 because ...". This applies only to requests about a specific order or a refund request, not to general policy questions such as how long refunds take.
        Never state an order status or a refund result unless a tool returned it.
        A refund is only REQUESTED until the customer confirms it with the Confirm button. Never say a refund is approved, completed or processed.
        """;

    public record Meta(String conversationId){}
    public record Token(String t){}
    public record Done(List<String> sources, boolean cached){}
    public record StreamError(String message){}

    private final ChatClient chatClient;
    private final OrderTool orderTool;
    private final RefundTool refundTool;
    private final Cache cache;
    private final ChatMemory chatMemory;
    private final ToolRouter tool;
    private final MeterRegistry meterRegistry;
    private final Counter hits;
    private final Counter misses;
    private final Counter skipped;
    private final Timer hitTimer;
    private final Timer modelTimer;

    public ChatController(ChatClient.Builder builder, ChatMemory chatMemory, VectorStore vec, OrderTool orderTool, RefundTool refundTool, Cache cache, ToolRouter t, MeterRegistry m){
        this.chatClient = builder.defaultSystem(p)
                .defaultAdvisors(MessageChatMemoryAdvisor.builder(chatMemory).build(),
                        QuestionAnswerAdvisor.builder(vec).searchRequest(SearchRequest.builder()
                                .topK(4).similarityThreshold(0.0).build()).build())
                .build();
        this.orderTool = orderTool;
        this.refundTool = refundTool;
        this.cache=cache;
        this.chatMemory = chatMemory;
        this.tool=t;
        this.meterRegistry = m;
        this.hits = Counter.builder("supportpilot.cache.requests").tag("result","hit").register(meterRegistry);
        this.misses = Counter.builder("supportpilot.cache.requests").tag("result","miss").register(meterRegistry);
        this.skipped = Counter.builder("supportpilot.cache.requests").tag("result","skipped").register(meterRegistry);
        this.hitTimer = Timer.builder("supportpilot.chat.latency").tag("served_by","cache").publishPercentiles(0.5,0.95).register(meterRegistry);
        this.modelTimer = Timer.builder("supportpilot.chat.latency").tag("served_by","model").publishPercentiles(0.5,0.95).register(meterRegistry);
        Gauge.builder("supportpilot.cache.hit.ratio", () -> {
            double hit = hits.count();double total = hit + misses.count();
            return total == 0 ? 0:hit/total;
        }).register(meterRegistry);
    }

    public static ServerSentEvent<Object> serverSentEvent(String event, Object data){
        return ServerSentEvent.<Object>builder().event(event).data(data).build();
    }

    @PostMapping("/chat")
    public ChatReply chat(@Valid @RequestBody ChatRequest request,
                          @RequestHeader("X-User-Id") String userId){
        Auth.checkValid(userId);/// checking if the userId is valid or not.
        String convId = giveId(request);
        /// this is the id corresponding to which conversation will be stored.
        String memoryId = userId + ":" + convId;
        String question = request.message();

        long start = System.nanoTime();

        boolean cacheable = cache.isCacheable(question) && chatMemory.get(memoryId).isEmpty();
        if(cacheable){
            var hit = cache.look(question);
            if(hit.isPresent()){
                hits.increment();
                var cached = hit.get();
                chatMemory.add(memoryId, List.of(new UserMessage(question),new AssistantMessage(cached.answer())));
                hitTimer.record(System.nanoTime() - start, TimeUnit.NANOSECONDS);
                return new ChatReply(cached.answer(), convId, cached.sources(), true);
            }
            misses.increment();
        }
        else{
            skipped.increment();
        }

        AtomicBoolean toolUsed = new AtomicBoolean(false);
        System.out.println("Before AI call, toolUsed = " + toolUsed.get());
        ToolRouter.Route route = tool.route(question);
        if(route == ToolRouter.Route.NONE && tool.hasOrderId(question)){
            String history = recentUserText(memoryId);
            if(!history.isBlank()){
                route = tool.route(history +" "+question);
            }
        }
        var promptSpec = chatClient.prompt()
                .user(request.message())
                .toolContext(Map.of("userId",userId, "toolUsed",toolUsed))
                .advisors(a->a.param(ChatMemory.CONVERSATION_ID, memoryId));
        switch (route) {
            case ORDER_STATUS ->
                    promptSpec = promptSpec.tools(orderTool);

            case REFUND_REQUEST ->
                    promptSpec = promptSpec.tools(refundTool);

            case NONE -> {
            }
        }
        ChatClientResponse response = promptSpec.call().chatClientResponse();

        String answer = response.chatResponse().getResult().getOutput().getText();
        List<String> sources = sourcesOf(response);

        System.out.println("cacheable = " + cacheable);
        System.out.println("toolUsed = " + toolUsed.get());
        System.out.println("sources = " + sources);
        System.out.println("answer is blank = "
                + (answer == null || answer.isBlank()));



        if(cacheable && !toolUsed.get() && !sources.isEmpty() && answer != null && !answer.isBlank()
                && !looksLikeNonAnswer(answer)){
            cache.store(question,answer,sources);
        }

        modelTimer.record(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        return new ChatReply(answer,convId,sources, false);
    }

    @PostMapping(value="/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public Flux<ServerSentEvent<Object>> stream(@Valid @RequestBody ChatRequest req,
                                               @RequestHeader("X-User-Id") String userId){
        //To check if the userId is valid or not
        Auth.checkValid(userId);
        String convId = giveId(req);
        String memoryId = userId +":"+convId;
        String question = req.message();
        long start = System.nanoTime();
        ServerSentEvent<Object> meta = serverSentEvent("meta",new Meta(convId));

        boolean cacheable = cache.isCacheable(question) && chatMemory.get(memoryId).isEmpty();
        if(cacheable){
            var hit = cache.look(question);
            if(hit.isPresent()){
                hits.increment();
                var cached = hit.get();
                chatMemory.add(memoryId, List.of(new UserMessage(question),new AssistantMessage(cached.answer())));
                hitTimer.record(System.nanoTime() - start, TimeUnit.NANOSECONDS);
                return Flux.just(meta, serverSentEvent("token", new Token(cached.answer())),serverSentEvent("done",new Done(cached.sources(), true)));
            }
            misses.increment();
        }
        else{
            skipped.increment();
        }
        StringBuilder answer = new StringBuilder();
        AtomicReference<List<String>> sources = new AtomicReference<>(List.of());
        Flux<ServerSentEvent<Object>> tokens = chatClient.prompt()
                .user(question)
                .advisors(a -> a.param(ChatMemory.CONVERSATION_ID, memoryId))
                .stream()
                .chatClientResponse()
                .doOnNext(r -> {
                    List<String> found = sourcesOf(r);
                    if(!found.isEmpty()) sources.set(found);
                })
                .map(this::textOf)
                .filter(t -> !t.isEmpty())
                .doOnNext(answer::append)
                .map(t -> serverSentEvent("token", new Token(t)));

        Flux<ServerSentEvent<Object>> finish = Flux.defer(() -> {
        String full = answer.toString();
        if(cacheable && !sources.get().isEmpty() && !full.isBlank() && !looksLikeNonAnswer(full)){
            cache.store(question, full, sources.get());
        }
        modelTimer.record(System.nanoTime() - start, TimeUnit.NANOSECONDS);
        return Flux.just(serverSentEvent("done", new Done(sources.get(), false)));});

        return Flux.concat(Flux.just(meta), tokens, finish)
                .onErrorResume(e -> Flux.just(serverSentEvent("error",
                        new StreamError("The assistant had a problem. Please try again."))));
    }

    private String textOf(ChatClientResponse r){
        var cr = r.chatResponse();
        if(cr == null || cr.getResult()==null || cr.getResult().getOutput() == null) return "";
        String t = cr.getResult().getOutput().getText();
        return t == null ? "" : t;
    }

    public String giveId(ChatRequest req){
        return req.conversationId()!=null ? req.conversationId() : UUID.randomUUID().toString();
    }
    private List<String> sourcesOf(ChatClientResponse res){
        System.out.println("Advisor context keys = " + res.context().keySet());

        Object docs = res.context().get(
                QuestionAnswerAdvisor.RETRIEVED_DOCUMENTS
        );

        System.out.println("Retrieved documents object = " + docs);

        if (docs == null) {
            System.out.println("Retrieved documents are NULL");
            return List.of();
        }

        if (!(docs instanceof List<?> list)) {
            System.out.println("Unexpected retrieved documents type = "
                    + docs.getClass().getName());
            return List.of();
        }

        System.out.println("Retrieved document count = " + list.size());

        return list.stream()
                .filter(Document.class::isInstance)
                .map(Document.class::cast)
                .map(d -> {
                    System.out.println("Document metadata = " + d.getMetadata());
                    return d.getMetadata().get("source");
                })
                .filter(java.util.Objects::nonNull)
                .map(Object::toString)
                .distinct()
                .toList();
    }

    private String recentUserText(String memoryId){
        List<String> texts = chatMemory.get(memoryId).stream()
                .filter(m->m.getMessageType() == MessageType.USER)
                .map(Message::getText)
                .toList();
        int from  = Math.max(0,texts.size()-3);
        return String.join(" ",texts.subList(from, texts.size()));
    }

    private boolean looksLikeNonAnswer(String a) {
        String s = a.toLowerCase();
        return s.contains("don't have") || s.contains("do not have")
                || s.contains("doesn't provide") || s.contains("does not provide")
                || s.contains("human agent");
    }
}
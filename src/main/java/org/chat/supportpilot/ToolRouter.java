package org.chat.supportpilot;

import org.springframework.stereotype.Component;

import java.util.Locale;
import java.util.regex.Pattern;

@Component
public class ToolRouter {

    public enum Route {
        NONE,
        ORDER_STATUS,
        REFUND_REQUEST
    }

    private static final Pattern ORDER_ID =
            Pattern.compile("\\bORD-\\d{1,10}\\b",
                    Pattern.CASE_INSENSITIVE);

    public Route route(String message) {
        String q = message.toLowerCase(Locale.ROOT);

        // 1. Detect an explicit request to initiate a refund.
        boolean refundIntent =
                q.contains("request a refund") || q.contains("request refund") ||
                        q.contains("initiate a refund") || q.contains("initiate refund") ||
                        q.contains("i want a refund") || q.contains("i'd like a refund") ||
                        q.contains("i would like a refund") || q.contains("please refund") ||
                        q.contains("refund my order") ||
                        (q.contains("refund") && (q.contains("i want") || q.contains("i need")
                                || q.contains("please") || q.contains("request")));

        boolean hasOrderId = ORDER_ID.matcher(message).find();
        boolean hasReason =
                q.contains("because ") ||
                        q.contains("reason:") ||
                        q.contains("damaged") ||
                        q.contains("defective") ||
                        q.contains("wrong item") ||
                        q.contains("not received") ||
                        q.contains("never arrived") ||
                        q.contains("faulty") ||
                        q.contains("broken");

        if (refundIntent && hasOrderId && hasReason) {
            return Route.REFUND_REQUEST;
        }
        boolean orderStatusIntent =
                q.contains("order status") ||
                        q.contains("track my order") ||
                        q.contains("track the order") ||
                        q.contains("where is my order") ||
                        q.contains("delivery status") ||
                        q.contains("order details") ||
                        q.contains("status of my order") ||
                        q.contains("status of order") || q.contains("status") || q.contains("track") || q.contains("where is") ||
                        q.contains("delivery");

        if (orderStatusIntent && hasOrderId) {
            return Route.ORDER_STATUS;
        }
        return Route.NONE;
    }
    public boolean hasOrderId(String message){
        return ORDER_ID.matcher(message).find();
    }
}

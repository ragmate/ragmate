 #!/bin/bash

API_URL="http://127.0.0.1:11434/api/checkout-event"
PREV_HEAD=$1
NEW_HEAD=$2

# Skip if nothing changed
if [ "$PREV_HEAD" = "$NEW_HEAD" ]; then
    exit 0
fi

# Construct JSON payload
JSON_PAYLOAD=$(jq -n \
                --arg prev "$PREV_HEAD" \
                --arg new "$NEW_HEAD" \
                '{previous_head: $prev, new_head: $new}')

# Make the API call
curl -s -o /dev/null -X POST \
     -H "Content-Type: application/json" \
     -d "$JSON_PAYLOAD" "$API_URL"

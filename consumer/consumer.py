import sys
import json
from confluent_kafka import Consumer
import csv

def handle_message(msg, filename):
    print("Message received.")
    topic = msg.topic()
    if topic != "raw_data" and topic != "predictions":
        return False
    
    parsed_msg = json.loads(msg.value().decode("iso-8859-1"))
    print(parsed_msg)
    with open(filename, "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        row = list()
        for key in parsed_msg.keys():
            row.append(parsed_msg[key])
        writer.writerows([row])
    return True


def main(args):
    running = True
    topic = args[1]
    filename = args[2]
    conf = {
        "bootstrap.servers": "kafka:9092",
        "auto.offset.reset": "smallest",
        "group.id": "tfd_consumers"
    }
    consumer = Consumer(conf)

    try:
        consumer.subscribe([topic])
        print(f"Kafka broker: {conf['bootstrap.servers']}")
        print(f"Successfully subscribed to topic: {topic}")

        print("Consumer loop started...")
        while running:
            msg = consumer.poll(timeout=3.0)
            if msg is None: continue
            if msg.error():
                print(msg.error())
            else:
                running = handle_message(msg, filename)
    finally:
        print("Closing kafka consumer...")
        consumer.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Missing parameters...")
        exit(1)
    main(sys.argv)
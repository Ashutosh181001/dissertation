import asyncio
import json
import websockets
from kafka import KafkaProducer
from datetime import datetime
import numpy as np

# Kafka config
KAFKA_TOPIC = "crypto_trades"
KAFKA_BROKER = "localhost:9092"
BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/btcusdt@trade"

# Kafka producer setup
producer = KafkaProducer(
    bootstrap_servers=KAFKA_BROKER,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

# Toggle anomaly injection
ENABLE_INJECTION = False
INJECTION_PROBABILITY = 0.01  # 1% chance

async def produce():
    async with websockets.connect(BINANCE_WS_URL) as websocket:
        print("Connected to Binance WebSocket.")
        while True:
            try:
                msg = await websocket.recv()
                data = json.loads(msg)

                price = float(data['p'])
                injected = ENABLE_INJECTION and np.random.rand() < INJECTION_PROBABILITY

                if injected:
                    price *= 1.05
                    print("⚠️ Injecting artificial anomaly at price:", round(price, 2))

                trade = {
                    "timestamp": datetime.utcfromtimestamp(data['T'] / 1000).isoformat(),
                    "price": price,
                    "quantity": float(data['q']),
                    "trade_id": data['t'],
                    "is_buyer_maker": data['m'],
                    "injected": injected  # <-- flag passed to consumer
                }

                print("Sending to Kafka:", trade)
                producer.send(KAFKA_TOPIC, value=trade)

            except Exception as e:
                print("Error:", e)
                await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(produce())
    except KeyboardInterrupt:
        print("Stopped producer.")

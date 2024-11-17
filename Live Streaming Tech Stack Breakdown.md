## Live Streaming Tech Stack Breakdown
Slide 1: Video Stream Ingestion

The initial step in a live streaming system involves capturing and encoding raw video input from various sources. This process requires efficient handling of video frames, audio synchronization, and proper buffering mechanisms to ensure smooth data flow through the pipeline.

```python
import cv2
import numpy as np
from queue import Queue
from threading import Thread

class VideoStreamIngestion:
    def __init__(self, source=0, buffer_size=64):
        self.stream = cv2.VideoCapture(source)
        self.buffer = Queue(maxsize=buffer_size)
        self.stopped = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        while True:
            if self.stopped:
                return
            if not self.buffer.full():
                ret, frame = self.stream.read()
                if not ret:
                    self.stop()
                    return
                self.buffer.put(frame)
                
    def read(self):
        return self.buffer.get()
        
    def stop(self):
        self.stopped = True
        self.stream.release()

# Example usage
stream = VideoStreamIngestion(source='rtsp://example.com/live/stream1').start()
while True:
    frame = stream.read()
    # Process frame here
    cv2.imshow('Live Stream', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
stream.stop()
```

Slide 2: Point-of-Presence Server Connection

Live streaming platforms implement a network of edge servers to minimize latency between streamers and the ingestion point. This implementation demonstrates establishing and managing connections to the nearest POP server using distance calculations and health checks.

```python
import requests
import geopy.distance
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class POPServer:
    id: str
    location: Tuple[float, float]
    endpoint: str
    health_status: bool = True
    
class POPServerManager:
    def __init__(self, pop_servers: List[POPServer]):
        self.pop_servers = pop_servers
        
    def find_nearest_healthy_pop(self, client_location: Tuple[float, float]) -> POPServer:
        available_pops = [pop for pop in self.pop_servers if pop.health_status]
        return min(available_pops, 
                  key=lambda pop: geopy.distance.distance(client_location, pop.location).km)
    
    def health_check(self, pop: POPServer) -> bool:
        try:
            response = requests.get(f"{pop.endpoint}/health", timeout=5)
            return response.status_code == 200
        except:
            return False

# Example usage
pop_servers = [
    POPServer("pop1", (40.7128, -74.0060), "https://pop1.stream.example.com"),
    POPServer("pop2", (51.5074, -0.1278), "https://pop2.stream.example.com"),
    POPServer("pop3", (35.6762, 139.6503), "https://pop3.stream.example.com"),
]

manager = POPServerManager(pop_servers)
client_location = (48.8566, 2.3522)  # Paris coordinates
nearest_pop = manager.find_nearest_healthy_pop(client_location)
print(f"Connected to POP: {nearest_pop.id}")
```

Slide 3: Video Transcoding Pipeline

The transcoding process converts the incoming video stream into multiple quality variants. This implementation showcases a pipeline that handles concurrent transcoding tasks using worker pools and monitors system resources to optimize performance.

```python
import multiprocessing
import ffmpeg
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class TranscodingProfile:
    resolution: str
    bitrate: str
    fps: int

class TranscodingPipeline:
    def __init__(self, worker_count: int = multiprocessing.cpu_count()):
        self.worker_count = worker_count
        self.profiles = {
            'high': TranscodingProfile('1920x1080', '6000k', 30),
            'medium': TranscodingProfile('1280x720', '2500k', 30),
            'low': TranscodingProfile('854x480', '1000k', 30)
        }
        
    def transcode_stream(self, input_url: str, output_path: str, profile: TranscodingProfile):
        try:
            stream = (
                ffmpeg
                .input(input_url)
                .output(
                    output_path,
                    vf=f'scale={profile.resolution}',
                    video_bitrate=profile.bitrate,
                    r=profile.fps,
                    format='hls',
                    hls_time=6,
                    hls_list_size=10
                )
                .overwrite_output()
                .run_async(pipe_stdout=True, pipe_stderr=True)
            )
            return stream
        except ffmpeg.Error as e:
            print(f"Transcoding error: {e.stderr.decode()}")
            return None

# Example usage
pipeline = TranscodingPipeline(worker_count=4)
input_stream = "rtmp://source.stream.example.com/live/stream1"

transcoding_tasks = []
for quality, profile in pipeline.profiles.items():
    output_path = f"/var/hls/{quality}/stream.m3u8"
    task = pipeline.transcode_stream(input_stream, output_path, profile)
    if task:
        transcoding_tasks.append(task)

# Wait for completion
for task in transcoding_tasks:
    task.wait()
```

Slide 4: HLS Packaging System

The HTTP Live Streaming (HLS) packaging system converts transcoded video segments into a standardized streaming format. This implementation handles manifest file generation, segment organization, and ensures proper timing for segment availability.

```python
import os
import time
from typing import List, Dict
from datetime import datetime

class HLSPackager:
    def __init__(self, base_path: str, segment_duration: int = 6):
        self.base_path = base_path
        self.segment_duration = segment_duration
        self.sequence_number = 0
        
    def generate_master_playlist(self, variants: List[Dict[str, str]]) -> str:
        playlist = "#EXTM3U\n#EXT-X-VERSION:3\n"
        for variant in variants:
            playlist += (f'#EXT-X-STREAM-INF:BANDWIDTH={variant["bandwidth"]},'
                       f'RESOLUTION={variant["resolution"]}\n{variant["path"]}\n')
        return playlist
    
    def generate_media_playlist(self, segments: List[str], target_duration: int) -> str:
        playlist = (f"#EXTM3U\n#EXT-X-VERSION:3\n"
                   f"#EXT-X-TARGETDURATION:{target_duration}\n"
                   f"#EXT-X-MEDIA-SEQUENCE:{self.sequence_number}\n")
        
        for segment in segments:
            playlist += f"#EXTINF:{self.segment_duration:.3f},\n{segment}\n"
        
        return playlist

    def package_stream(self, input_segments: List[str], quality: str):
        output_dir = os.path.join(self.base_path, quality)
        os.makedirs(output_dir, exist_ok=True)
        
        segments = []
        for segment in input_segments:
            segment_name = f"segment_{self.sequence_number}.ts"
            output_path = os.path.join(output_dir, segment_name)
            # Copy segment to output location
            with open(segment, 'rb') as src, open(output_path, 'wb') as dst:
                dst.write(src.read())
            segments.append(segment_name)
            self.sequence_number += 1
        
        return segments

# Example usage
packager = HLSPackager("/var/www/hls")

variants = [
    {"bandwidth": "6000000", "resolution": "1920x1080", "path": "high/playlist.m3u8"},
    {"bandwidth": "2500000", "resolution": "1280x720", "path": "medium/playlist.m3u8"},
    {"bandwidth": "1000000", "resolution": "854x480", "path": "low/playlist.m3u8"}
]

master_playlist = packager.generate_master_playlist(variants)
with open(os.path.join(packager.base_path, "master.m3u8"), "w") as f:
    f.write(master_playlist)

# Package segments for each quality variant
input_segments = [f"/tmp/transcoded/high/segment_{i}.ts" for i in range(5)]
packaged_segments = packager.package_stream(input_segments, "high")
media_playlist = packager.generate_media_playlist(packaged_segments, 6)

with open(os.path.join(packager.base_path, "high/playlist.m3u8"), "w") as f:
    f.write(media_playlist)
```

Slide 5: CDN Integration Layer

Content Delivery Network integration is crucial for distributing live stream content globally. This implementation demonstrates how to manage CDN edge caching, purging, and load balancing across multiple CDN providers.

```python
import aiohttp
import asyncio
from typing import List, Dict
from dataclasses import dataclass
from enum import Enum

class CDNProvider(Enum):
    CLOUDFLARE = "cloudflare"
    AKAMAI = "akamai"
    FASTLY = "fastly"

@dataclass
class CDNConfig:
    provider: CDNProvider
    api_key: str
    zone_id: str
    endpoints: List[str]

class CDNManager:
    def __init__(self, configs: List[CDNConfig]):
        self.configs = configs
        self.session = None
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        
    async def close(self):
        if self.session:
            await self.session.close()
            
    async def purge_cache(self, urls: List[str], provider: CDNProvider):
        config = next(c for c in self.configs if c.provider == provider)
        headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
        
        if provider == CDNProvider.CLOUDFLARE:
            endpoint = f"https://api.cloudflare.com/client/v4/zones/{config.zone_id}/purge_cache"
            data = {"files": urls}
            
        async with self.session.post(endpoint, json=data, headers=headers) as response:
            return await response.json()
            
    async def health_check(self) -> Dict[str, bool]:
        tasks = []
        for config in self.configs:
            for endpoint in config.endpoints:
                tasks.append(self.check_endpoint(endpoint))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return dict(zip([f"{c.provider.value}-{e}" for c in self.configs 
                        for e in c.endpoints], results))
                
    async def check_endpoint(self, endpoint: str) -> bool:
        try:
            async with self.session.get(endpoint, timeout=5) as response:
                return response.status == 200
        except:
            return False

# Example usage
async def main():
    cdn_configs = [
        CDNConfig(
            provider=CDNProvider.CLOUDFLARE,
            api_key="your_api_key",
            zone_id="your_zone_id",
            endpoints=["https://edge1.cdn.example.com", "https://edge2.cdn.example.com"]
        )
    ]
    
    cdn_manager = CDNManager(cdn_configs)
    await cdn_manager.initialize()
    
    # Check CDN health
    health_status = await cdn_manager.health_check()
    print("CDN Health Status:", health_status)
    
    # Purge cache for specific URLs
    urls_to_purge = [
        "https://stream.example.com/hls/high/segment_1.ts",
        "https://stream.example.com/hls/high/playlist.m3u8"
    ]
    
    result = await cdn_manager.purge_cache(urls_to_purge, CDNProvider.CLOUDFLARE)
    print("Cache Purge Result:", result)
    
    await cdn_manager.close()

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 6: Video Player Integration

The player integration layer handles video playback, adaptive bitrate switching, and buffer management. This implementation shows how to create a custom video player that supports HLS playback with quality adaptation.

```python
import asyncio
import m3u8
import aiohttp
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class StreamQuality:
    bandwidth: int
    resolution: str
    url: str
    
class AdaptivePlayer:
    def __init__(self, master_playlist_url: str, buffer_size: int = 30):
        self.master_url = master_playlist_url
        self.buffer_size = buffer_size
        self.current_quality: Optional[StreamQuality] = None
        self.buffer: List[str] = []
        self.session = None
        
    async def initialize(self):
        self.session = aiohttp.ClientSession()
        await self.load_master_playlist()
        
    async def load_master_playlist(self):
        async with self.session.get(self.master_url) as response:
            content = await response.text()
            playlist = m3u8.loads(content)
            
            self.qualities = [
                StreamQuality(
                    bandwidth=stream.bandwidth,
                    resolution=f"{stream.resolution[0]}x{stream.resolution[1]}",
                    url=stream.uri
                )
                for stream in playlist.playlists
            ]
            
            # Start with highest quality
            self.current_quality = max(self.qualities, key=lambda q: q.bandwidth)
            
    async def monitor_bandwidth(self):
        while True:
            start_time = asyncio.get_event_loop().time()
            async with self.session.get(self.current_quality.url) as response:
                content = await response.read()
                
            download_time = asyncio.get_event_loop().time() - start_time
            measured_bandwidth = len(content) * 8 / download_time  # bits per second
            
            # Adjust quality based on available bandwidth
            available_qualities = [q for q in self.qualities 
                                 if q.bandwidth < measured_bandwidth * 0.8]
            if available_qualities:
                new_quality = max(available_qualities, key=lambda q: q.bandwidth)
                if new_quality != self.current_quality:
                    print(f"Switching quality: {self.current_quality.resolution} "
                          f"-> {new_quality.resolution}")
                    self.current_quality = new_quality
            
            await asyncio.sleep(5)
            
    async def buffer_segments(self):
        while True:
            if len(self.buffer) < self.buffer_size:
                async with self.session.get(self.current_quality.url) as response:
                    content = await response.text()
                    playlist = m3u8.loads(content)
                    
                    for segment in playlist.segments:
                        if segment.uri not in self.buffer:
                            self.buffer.append(segment.uri)
                            
                    while len(self.buffer) > self.buffer_size:
                        self.buffer.pop(0)
                        
            await asyncio.sleep(1)

# Example usage
async def main():
    player = AdaptivePlayer("https://stream.example.com/master.m3u8")
    await player.initialize()
    
    # Start bandwidth monitoring and buffer management
    await asyncio.gather(
        player.monitor_bandwidth(),
        player.buffer_segments()
    )

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 7: Video Storage and Replay System

The storage and replay system handles archiving live streams for video-on-demand access. This implementation demonstrates efficient storage management, segment organization, and replay functionality using cloud storage services.

```python
import boto3
import asyncio
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path

class StreamArchiveManager:
    def __init__(self, bucket_name: str, retention_days: int = 30):
        self.s3 = boto3.client('s3')
        self.bucket = bucket_name
        self.retention_days = retention_days
        
    async def archive_stream_segment(self, stream_id: str, segment_path: str):
        timestamp = datetime.now().isoformat()
        s3_key = f"archives/{stream_id}/{timestamp}/{Path(segment_path).name}"
        
        with open(segment_path, 'rb') as segment_file:
            self.s3.upload_fileobj(
                segment_file,
                self.bucket,
                s3_key,
                ExtraArgs={
                    'Metadata': {
                        'stream_id': stream_id,
                        'timestamp': timestamp,
                        'type': 'hls-segment'
                    }
                }
            )
        return s3_key
        
    async def create_replay_manifest(self, stream_id: str, 
                                   start_time: datetime, 
                                   end_time: datetime) -> str:
        # List all segments for the stream within time range
        segments = self.s3.list_objects_v2(
            Bucket=self.bucket,
            Prefix=f"archives/{stream_id}/"
        ).get('Contents', [])
        
        relevant_segments = [
            s for s in segments
            if start_time <= datetime.fromisoformat(
                s['Metadata']['timestamp']) <= end_time
        ]
        
        # Generate HLS manifest
        manifest = "#EXTM3U\n#EXT-X-VERSION:3\n#EXT-X-PLAYLIST-TYPE:VOD\n"
        manifest += f"#EXT-X-TARGETDURATION:6\n"
        
        for segment in sorted(relevant_segments, 
                            key=lambda x: x['Metadata']['timestamp']):
            manifest += f"#EXTINF:6.0,\n{self.get_presigned_url(segment['Key'])}\n"
        
        manifest += "#EXT-X-ENDLIST"
        
        # Save manifest to S3
        manifest_key = f"replays/{stream_id}/{start_time.isoformat()}_manifest.m3u8"
        self.s3.put_object(
            Bucket=self.bucket,
            Key=manifest_key,
            Body=manifest.encode('utf-8'),
            ContentType='application/x-mpegURL'
        )
        
        return self.get_presigned_url(manifest_key)
    
    def get_presigned_url(self, key: str, expires: int = 3600) -> str:
        return self.s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': self.bucket, 'Key': key},
            ExpiresIn=expires
        )
        
    async def cleanup_old_archives(self):
        while True:
            cutoff_date = datetime.now() - timedelta(days=self.retention_days)
            
            paginator = self.s3.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=self.bucket, Prefix='archives/'):
                for obj in page.get('Contents', []):
                    timestamp = datetime.fromisoformat(
                        obj['Metadata']['timestamp']
                    )
                    if timestamp < cutoff_date:
                        self.s3.delete_object(
                            Bucket=self.bucket,
                            Key=obj['Key']
                        )
            
            await asyncio.sleep(86400)  # Run daily

# Example usage
async def main():
    archive_manager = StreamArchiveManager('my-streaming-bucket')
    
    # Archive new segment
    segment_key = await archive_manager.archive_stream_segment(
        'stream123',
        '/tmp/segments/segment_001.ts'
    )
    print(f"Archived segment: {segment_key}")
    
    # Create replay manifest
    start_time = datetime.now() - timedelta(hours=1)
    end_time = datetime.now()
    replay_url = await archive_manager.create_replay_manifest(
        'stream123',
        start_time,
        end_time
    )
    print(f"Replay URL: {replay_url}")
    
    # Start cleanup task
    asyncio.create_task(archive_manager.cleanup_old_archives())

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 8: Analytics and Monitoring System

A comprehensive analytics system tracking viewer engagement, stream health, and performance metrics. This implementation showcases real-time metrics collection, aggregation, and alerting capabilities.

```python
import time
import redis
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

@dataclass
class StreamMetrics:
    stream_id: str
    viewers: int
    buffer_health: float
    bitrate: int
    latency: float
    errors: int
    timestamp: datetime

class StreamAnalytics:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port)
        self.logger = logging.getLogger(__name__)
        
    def record_metrics(self, metrics: StreamMetrics):
        key = f"stream:{metrics.stream_id}:metrics:{metrics.timestamp.isoformat()}"
        self.redis.hmset(key, {
            'viewers': metrics.viewers,
            'buffer_health': metrics.buffer_health,
            'bitrate': metrics.bitrate,
            'latency': metrics.latency,
            'errors': metrics.errors
        })
        self.redis.expire(key, 86400)  # Keep metrics for 24 hours
        
        self._check_alerts(metrics)
        
    def _check_alerts(self, metrics: StreamMetrics):
        alerts = []
        
        if metrics.buffer_health < 0.5:
            alerts.append(f"Low buffer health: {metrics.buffer_health:.2f}")
        
        if metrics.latency > 10.0:
            alerts.append(f"High latency: {metrics.latency:.2f}s")
            
        if metrics.errors > 100:
            alerts.append(f"High error rate: {metrics.errors} errors")
            
        if alerts:
            self.logger.warning(f"Stream {metrics.stream_id} alerts: {', '.join(alerts)}")
            
    def get_stream_health(self, stream_id: str, 
                         time_window: timedelta = timedelta(minutes=5)) -> Dict:
        start_time = datetime.now() - time_window
        keys = self.redis.keys(f"stream:{stream_id}:metrics:*")
        
        metrics = []
        for key in keys:
            timestamp = datetime.fromisoformat(key.decode().split(':')[-1])
            if timestamp >= start_time:
                data = self.redis.hgetall(key)
                metrics.append({
                    'timestamp': timestamp,
                    **{k.decode(): float(v) for k, v in data.items()}
                })
                
        if not metrics:
            return {}
            
        return {
            'average_viewers': sum(m['viewers'] for m in metrics) / len(metrics),
            'average_latency': sum(m['latency'] for m in metrics) / len(metrics),
            'total_errors': sum(m['errors'] for m in metrics),
            'min_buffer_health': min(m['buffer_health'] for m in metrics),
            'max_bitrate': max(m['bitrate'] for m in metrics)
        }

# Example usage
analytics = StreamAnalytics()

# Record metrics
metrics = StreamMetrics(
    stream_id='stream123',
    viewers=1000,
    buffer_health=0.8,
    bitrate=2500000,
    latency=2.5,
    errors=5,
    timestamp=datetime.now()
)
analytics.record_metrics(metrics)

# Get health report
health_report = analytics.get_stream_health('stream123')
print("Stream Health Report:", health_report)
```

Slide 9: Error Handling and Recovery System

A robust error handling system is crucial for maintaining stream stability. This implementation demonstrates comprehensive error detection, automatic recovery procedures, and failover mechanisms for live streaming components.

```python
import asyncio
import logging
from enum import Enum
from typing import Dict, Optional, Callable
from datetime import datetime, timedelta

class StreamError(Enum):
    INGESTION_FAILED = "ingestion_failed"
    TRANSCODING_ERROR = "transcoding_error"
    PACKAGING_ERROR = "packaging_error"
    CDN_ERROR = "cdn_error"
    NETWORK_ERROR = "network_error"

class StreamRecoverySystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_counts: Dict[str, int] = {}
        self.recovery_handlers: Dict[StreamError, Callable] = {}
        self.last_recovery_attempt: Dict[str, datetime] = {}
        
    def register_recovery_handler(self, 
                                error_type: StreamError, 
                                handler: Callable):
        self.recovery_handlers[error_type] = handler
        
    async def handle_error(self, 
                          stream_id: str, 
                          error_type: StreamError, 
                          error_data: Dict):
        self.error_counts.setdefault(stream_id, 0)
        self.error_counts[stream_id] += 1
        
        self.logger.error(f"Stream {stream_id} encountered {error_type.value}: {error_data}")
        
        # Check if we should attempt recovery
        if await self._should_attempt_recovery(stream_id, error_type):
            await self._execute_recovery(stream_id, error_type, error_data)
            
    async def _should_attempt_recovery(self, 
                                     stream_id: str, 
                                     error_type: StreamError) -> bool:
        # Don't retry too frequently
        last_attempt = self.last_recovery_attempt.get(stream_id)
        if last_attempt and datetime.now() - last_attempt < timedelta(minutes=5):
            return False
            
        # Don't retry too many times
        if self.error_counts[stream_id] > 3:
            self.logger.critical(
                f"Stream {stream_id} exceeded maximum recovery attempts"
            )
            return False
            
        return True
        
    async def _execute_recovery(self, 
                              stream_id: str, 
                              error_type: StreamError, 
                              error_data: Dict):
        self.last_recovery_attempt[stream_id] = datetime.now()
        
        handler = self.recovery_handlers.get(error_type)
        if not handler:
            self.logger.warning(f"No recovery handler for {error_type.value}")
            return
            
        try:
            await handler(stream_id, error_data)
            self.logger.info(f"Recovery successful for stream {stream_id}")
            self.error_counts[stream_id] = 0
        except Exception as e:
            self.logger.error(f"Recovery failed for stream {stream_id}: {str(e)}")
            
class StreamFailoverManager:
    def __init__(self, backup_endpoints: Dict[str, str]):
        self.backup_endpoints = backup_endpoints
        self.active_failovers: Dict[str, str] = {}
        
    async def initiate_failover(self, stream_id: str) -> Optional[str]:
        if stream_id in self.active_failovers:
            return None
            
        backup_endpoint = self.backup_endpoints.get(stream_id)
        if not backup_endpoint:
            return None
            
        self.active_failovers[stream_id] = backup_endpoint
        return backup_endpoint
        
    async def revert_failover(self, stream_id: str):
        if stream_id in self.active_failovers:
            del self.active_failovers[stream_id]

# Example usage
async def main():
    # Initialize recovery system
    recovery_system = StreamRecoverySystem()
    failover_manager = StreamFailoverManager({
        'stream123': 'rtmp://backup.example.com/live'
    })
    
    # Define recovery handlers
    async def handle_ingestion_error(stream_id: str, error_data: Dict):
        backup_endpoint = await failover_manager.initiate_failover(stream_id)
        if backup_endpoint:
            print(f"Failing over stream {stream_id} to {backup_endpoint}")
            # Implementation of failover logic here
            
    async def handle_transcoding_error(stream_id: str, error_data: Dict):
        print(f"Restarting transcoding pipeline for stream {stream_id}")
        # Implementation of transcoding restart logic here
        
    # Register recovery handlers
    recovery_system.register_recovery_handler(
        StreamError.INGESTION_FAILED, 
        handle_ingestion_error
    )
    recovery_system.register_recovery_handler(
        StreamError.TRANSCODING_ERROR, 
        handle_transcoding_error
    )
    
    # Simulate error scenarios
    await recovery_system.handle_error(
        'stream123',
        StreamError.INGESTION_FAILED,
        {'reason': 'Connection timeout'}
    )
    
    await recovery_system.handle_error(
        'stream123',
        StreamError.TRANSCODING_ERROR,
        {'reason': 'GPU memory exhausted'}
    )

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 10: Stream Scaling and Load Balancing

This implementation manages the dynamic scaling of streaming infrastructure based on load patterns and viewer distribution, ensuring optimal resource utilization and performance.

```python
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
from datetime import datetime

class ServerType(Enum):
    INGEST = "ingest"
    TRANSCODE = "transcode"
    EDGE = "edge"

@dataclass
class ServerInstance:
    id: str
    type: ServerType
    capacity: int
    current_load: int
    region: str
    status: str
    startup_time: datetime

class AutoScaler:
    def __init__(self, 
                 target_utilization: float = 0.7,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3):
        self.servers: Dict[str, ServerInstance] = {}
        self.target_utilization = target_utilization
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        
    async def monitor_load(self):
        while True:
            await self._check_scaling_needs()
            await asyncio.sleep(60)  # Check every minute
            
    async def _check_scaling_needs(self):
        for server_type in ServerType:
            servers = self._get_servers_by_type(server_type)
            if not servers:
                continue
                
            avg_utilization = sum(s.current_load / s.capacity 
                                for s in servers) / len(servers)
            
            if avg_utilization > self.scale_up_threshold:
                await self._scale_up(server_type)
            elif avg_utilization < self.scale_down_threshold:
                await self._scale_down(server_type)
                
    async def _scale_up(self, server_type: ServerType):
        new_capacity_needed = self._calculate_new_capacity(server_type)
        if new_capacity_needed <= 0:
            return
            
        new_instance = await self._provision_server(server_type)
        if new_instance:
            self.servers[new_instance.id] = new_instance
            print(f"Scaled up {server_type.value} server: {new_instance.id}")
            
    async def _scale_down(self, server_type: ServerType):
        servers = self._get_servers_by_type(server_type)
        if len(servers) <= 1:
            return  # Keep at least one server
            
        # Find least loaded server
        server_to_remove = min(servers, key=lambda s: s.current_load)
        await self._decommission_server(server_to_remove.id)
        del self.servers[server_to_remove.id]
        print(f"Scaled down {server_type.value} server: {server_to_remove.id}")
        
    def _calculate_new_capacity(self, server_type: ServerType) -> int:
        servers = self._get_servers_by_type(server_type)
        total_capacity = sum(s.capacity for s in servers)
        total_load = sum(s.current_load for s in servers)
        
        target_capacity = total_load / self.target_utilization
        return max(0, int(target_capacity - total_capacity))
        
    async def _provision_server(self, 
                              server_type: ServerType) -> Optional[ServerInstance]:
        # Simulated server provisioning
        server_id = f"{server_type.value}-{len(self.servers) + 1}"
        return ServerInstance(
            id=server_id,
            type=server_type,
            capacity=1000,
            current_load=0,
            region="us-east-1",
            status="active",
            startup_time=datetime.now()
        )
        
    async def _decommission_server(self, server_id: str):
        # Implement graceful shutdown logic here
        pass
        
    def _get_servers_by_type(self, server_type: ServerType) -> List[ServerInstance]:
        return [s for s in self.servers.values() if s.type == server_type]

# Example usage
async def main():
    scaler = AutoScaler()
    
    # Add initial servers
    initial_servers = [
        ServerInstance("ingest-1", ServerType.INGEST, 1000, 600, 
                      "us-east-1", "active", datetime.now()),
        ServerInstance("transcode-1", ServerType.TRANSCODE, 1000, 800, 
                      "us-east-1", "active", datetime.now())
    ]
    
    for server in initial_servers:
        scaler.servers[server.id] = server
    
    # Start monitoring
    await scaler.monitor_load()

if __name__ == "__main__":
    asyncio.run(main())
```

Slide 11: Stream Quality Assessment System

This implementation provides real-time quality assessment of video streams, measuring parameters like bitrate stability, frame drops, and audio synchronization to ensure optimal viewing experience.

```python
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque
import cv2

@dataclass
class QualityMetrics:
    psnr: float
    bitrate_stability: float
    frame_drop_rate: float
    audio_sync_offset: float
    buffering_ratio: float
    
class StreamQualityAnalyzer:
    def __init__(self, window_size: int = 300):  # 5 minutes at 60fps
        self.frame_buffer = deque(maxlen=window_size)
        self.bitrate_history = deque(maxlen=window_size)
        self.frame_timestamps = deque(maxlen=window_size)
        self.audio_timestamps = deque(maxlen=window_size)
        
    def analyze_frame(self, 
                     frame: np.ndarray, 
                     frame_size: int, 
                     timestamp: float,
                     audio_timestamp: float) -> QualityMetrics:
        # Store frame data
        self.frame_buffer.append(frame)
        self.bitrate_history.append(frame_size)
        self.frame_timestamps.append(timestamp)
        self.audio_timestamps.append(audio_timestamp)
        
        return QualityMetrics(
            psnr=self._calculate_psnr(),
            bitrate_stability=self._calculate_bitrate_stability(),
            frame_drop_rate=self._calculate_frame_drop_rate(),
            audio_sync_offset=self._calculate_audio_sync(),
            buffering_ratio=self._calculate_buffering_ratio()
        )
        
    def _calculate_psnr(self) -> float:
        if len(self.frame_buffer) < 2:
            return float('inf')
            
        current_frame = self.frame_buffer[-1]
        previous_frame = self.frame_buffer[-2]
        
        mse = np.mean((current_frame - previous_frame) ** 2)
        if mse == 0:
            return float('inf')
            
        max_pixel = 255.0
        return 20 * np.log10(max_pixel / np.sqrt(mse))
        
    def _calculate_bitrate_stability(self) -> float:
        if len(self.bitrate_history) < 2:
            return 1.0
            
        bitrates = np.array(self.bitrate_history)
        stability = 1.0 - (np.std(bitrates) / np.mean(bitrates))
        return max(0.0, min(1.0, stability))
        
    def _calculate_frame_drop_rate(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
            
        timestamps = np.array(self.frame_timestamps)
        frame_intervals = np.diff(timestamps)
        expected_interval = 1.0 / 60  # Assuming 60fps
        
        drops = np.sum(frame_intervals > (expected_interval * 1.5))
        return drops / len(frame_intervals)
        
    def _calculate_audio_sync(self) -> float:
        if not self.frame_timestamps or not self.audio_timestamps:
            return 0.0
            
        video_ts = np.array(self.frame_timestamps)
        audio_ts = np.array(self.audio_timestamps)
        
        return np.mean(np.abs(video_ts - audio_ts))
        
    def _calculate_buffering_ratio(self) -> float:
        if len(self.frame_timestamps) < 2:
            return 0.0
            
        total_time = self.frame_timestamps[-1] - self.frame_timestamps[0]
        expected_frames = total_time * 60  # Assuming 60fps
        actual_frames = len(self.frame_timestamps)
        
        return max(0.0, min(1.0, actual_frames / expected_frames))
        
class QualityController:
    def __init__(self, target_quality: float = 0.8):
        self.analyzer = StreamQualityAnalyzer()
        self.target_quality = target_quality
        
    def process_frame(self, 
                     frame: np.ndarray, 
                     audio_data: np.ndarray) -> Dict[str, float]:
        frame_size = frame.nbytes
        current_time = time.time()
        
        # Analyze frame quality
        metrics = self.analyzer.analyze_frame(
            frame,
            frame_size,
            current_time,
            current_time + self._estimate_audio_delay(audio_data)
        )
        
        # Generate quality report
        quality_report = {
            'psnr': metrics.psnr,
            'bitrate_stability': metrics.bitrate_stability,
            'frame_drop_rate': metrics.frame_drop_rate,
            'audio_sync': metrics.audio_sync_offset,
            'buffering_ratio': metrics.buffering_ratio,
            'overall_quality': self._calculate_overall_quality(metrics)
        }
        
        return quality_report
        
    def _estimate_audio_delay(self, audio_data: np.ndarray) -> float:
        # Simplified audio delay estimation
        return len(audio_data) / 44100  # Assuming 44.1kHz sample rate
        
    def _calculate_overall_quality(self, metrics: QualityMetrics) -> float:
        weights = {
            'psnr': 0.3,
            'bitrate_stability': 0.2,
            'frame_drop_rate': 0.2,
            'audio_sync': 0.15,
            'buffering_ratio': 0.15
        }
        
        # Normalize PSNR to 0-1 range (assuming typical PSNR range of 20-50)
        normalized_psnr = min(1.0, max(0.0, (metrics.psnr - 20) / 30))
        
        quality_score = (
            weights['psnr'] * normalized_psnr +
            weights['bitrate_stability'] * metrics.bitrate_stability +
            weights['frame_drop_rate'] * (1 - metrics.frame_drop_rate) +
            weights['audio_sync'] * (1 - min(1.0, metrics.audio_sync_offset / 0.5)) +
            weights['buffering_ratio'] * metrics.buffering_ratio
        )
        
        return quality_score

# Example usage
quality_controller = QualityController()

# Simulate frame processing
cap = cv2.VideoCapture(0)
ret, frame = cap.read()

if ret:
    # Simulate audio data (random samples)
    audio_data = np.random.rand(44100)  # 1 second of audio at 44.1kHz
    
    quality_report = quality_controller.process_frame(frame, audio_data)
    print("Stream Quality Report:", quality_report)

cap.release()
```

Slide 12: Stream Authentication and Security

This implementation provides comprehensive security measures for live streaming, including token-based authentication, stream encryption, and access control management.

```python
import jwt
import hashlib
import secrets
from typing import Dict, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from dataclasses import dataclass

@dataclass
class StreamCredentials:
    stream_key: str
    token: str
    expiry: datetime

class StreamSecurityManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.fernet = Fernet(Fernet.generate_key())
        self.active_streams: Dict[str, StreamCredentials] = {}
        
    def generate_stream_credentials(self, 
                                  user_id: str, 
                                  stream_id: str,
                                  duration: timedelta = timedelta(hours=4)) -> StreamCredentials:
        # Generate unique stream key
        stream_key = self._generate_stream_key()
        
        # Generate JWT token
        expiry = datetime.utcnow() + duration
        token = jwt.encode({
            'user_id': user_id,
            'stream_id': stream_id,
            'stream_key': stream_key,
            'exp': expiry
        }, self.secret_key, algorithm='HS256')
        
        credentials = StreamCredentials(
            stream_key=stream_key,
            token=token,
            expiry=expiry
        )
        
        self.active_streams[stream_id] = credentials
        return credentials
        
    def validate_stream_access(self, 
                             stream_id: str, 
                             token: str) -> bool:
        try:
            # Verify JWT token
            payload = jwt.decode(token, 
                               self.secret_key, 
                               algorithms=['HS256'])
            
            # Check if stream exists and matches
            if stream_id != payload['stream_id']:
                return False
                
            # Check if stream is active
            if stream_id not in self.active_streams:
                return False
                
            # Check expiration
            if datetime.utcnow() > self.active_streams[stream_id].expiry:
                del self.active_streams[stream_id]
                return False
                
            return True
            
        except jwt.InvalidTokenError:
            return False
            
    def encrypt_stream_segment(self, 
                             segment_data: bytes, 
                             stream_id: str) -> Optional[bytes]:
        if stream_id not in self.active_streams:
            return None
            
        return self.fernet.encrypt(segment_data)
        
    def decrypt_stream_segment(self, 
                             encrypted_data: bytes, 
                             stream_id: str) -> Optional[bytes]:
        if stream_id not in self.active_streams:
            return None
            
        return self.fernet.decrypt(encrypted_data)
        
    def revoke_stream_access(self, stream_id: str):
        if stream_id in self.active_streams:
            del self.active_streams[stream_id]
            
    def _generate_stream_key(self, length: int = 32) -> str:
        return secrets.token_urlsafe(length)
        
class AccessControlList:
    def __init__(self):
        self.stream_permissions: Dict[str, Dict[str, bool]] = {}
        
    def grant_access(self, stream_id: str, user_id: str):
        if stream_id not in self.stream_permissions:
            self.stream_permissions[stream_id] = {}
        self.stream_permissions[stream_id][user_id] = True
        
    def revoke_access(self, stream_id: str, user_id: str):
        if stream_id in self.stream_permissions:
            self.stream_permissions[stream_id].pop(user_id, None)
            
    def check_access(self, stream_id: str, user_id: str) -> bool:
        return (stream_id in self.stream_permissions and
                user_id in self.stream_permissions[stream_id] and
                self.stream_permissions[stream_id][user_id])

# Example usage
def main():
    # Initialize security manager
    security_manager = StreamSecurityManager('your-secret-key')
    acl = AccessControlList()
    
    # Generate stream credentials
    user_id = "user123"
    stream_id = "stream456"
    
    credentials = security_manager.generate_stream_credentials(
        user_id,
        stream_id
    )
    print(f"Stream Credentials: {credentials}")
    
    # Grant access
    acl.grant_access(stream_id, user_id)
    
    # Validate access
    has_access = (
        acl.check_access(stream_id, user_id) and
        security_manager.validate_stream_access(stream_id, credentials.token)
    )
    print(f"Access Validated: {has_access}")
    
    # Simulate stream encryption
    original_data = b"Sample stream segment data"
    encrypted_data = security_manager.encrypt_stream_segment(
        original_data,
        stream_id
    )
    
    # Simulate stream decryption
    decrypted_data = security_manager.decrypt_stream_segment(
        encrypted_data,
        stream_id
    )
    print(f"Decryption successful: {original_data == decrypted_data}")
    
    # Revoke access
    security_manager.revoke_stream_access(stream_id)
    acl.revoke_access(stream_id, user_id)

if __name__ == "__main__":
    main()
```

Slide 13: Additional Resources

*   Real-time Video Content Analysis with Deep Learning
    *   [https://arxiv.org/abs/2103.06838](https://arxiv.org/abs/2103.06838)
*   Adaptive Bitrate Streaming: A Survey and Future Research Directions
    *   [https://arxiv.org/abs/2102.07108](https://arxiv.org/abs/2102.07108)
*   Low-Latency Live Streaming: State of the Art and Future Directions
    *   [https://dl.acm.org/doi/10.1145/3339825.3391858](https://dl.acm.org/doi/10.1145/3339825.3391858)
*   Optimal Resource Allocation for Live Video Streaming Systems
    *   [https://www.computer.org/csdl/journal/mm/2023/02/09714037/1A7NXBQqi4g](https://www.computer.org/csdl/journal/mm/2023/02/09714037/1A7NXBQqi4g)
*   Deep Learning for Video Streaming Quality Assessment
    *   For detailed information on this topic, search for "QoE assessment in video streaming using deep learning"
*   Real-time Video Processing Optimization Techniques
    *   Recommended search: "GPU acceleration for real-time video processing"
*   Content Delivery Networks for Live Streaming
    *   For implementation details, search for "CDN architecture for live video delivery"


#!/usr/bin/env python3
"""
Script para gerar Parquet de metadados do dataset de vídeos no S3.

Isso acelera MUITO o startup do SimpleTuner para datasets massivos,
evitando que ele precise escanear cada arquivo individualmente.

Uso:
    python generate_metadata_parquet.py \
        --bucket YOUR_BUCKET_NAME \
        --prefix "videos/" \
        --output metadata.parquet \
        --region us-east-1 \
        --workers 32
"""

import argparse
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import cv2
import pandas as pd
from tqdm import tqdm


def get_s3_client(region: str, endpoint_url: str = None):
    """Cria cliente S3."""
    return boto3.client(
        "s3",
        region_name=region,
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )


def list_s3_videos(client, bucket: str, prefix: str) -> list[str]:
    """Lista todos os arquivos .mp4 no bucket S3."""
    videos = []
    paginator = client.get_paginator("list_objects_v2")

    print(f"Listando vídeos em s3://{bucket}/{prefix}...")

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.lower().endswith((".mp4", ".mov", ".avi", ".webm")):
                videos.append(key)

    print(f"Encontrados {len(videos)} vídeos")
    return videos


def get_video_metadata(client, bucket: str, video_key: str) -> dict | None:
    """Extrai metadados de um vídeo (baixa temporariamente)."""
    try:
        # Baixa o vídeo para um arquivo temporário
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            client.download_file(bucket, video_key, tmp.name)

            # Abre com OpenCV para extrair metadados
            cap = cv2.VideoCapture(tmp.name)
            if not cap.isOpened():
                return None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

        # Tenta ler a caption do arquivo .txt correspondente
        caption = ""
        txt_key = Path(video_key).with_suffix(".txt").as_posix()
        try:
            response = client.get_object(Bucket=bucket, Key=txt_key)
            caption = response["Body"].read().decode("utf-8").strip()
            # Pega apenas a primeira linha se houver múltiplas
            caption = caption.split("\n")[0].strip()
        except Exception:
            # Se não encontrar o .txt, usa o nome do arquivo
            caption = Path(video_key).stem.replace("_", " ").replace("-", " ")

        return {
            "filename": Path(video_key).name,
            "s3_key": video_key,
            "caption": caption,
            "width": width,
            "height": height,
            "fps": fps,
            "num_frames": frame_count,
            "duration_seconds": round(duration, 2),
        }

    except Exception as e:
        print(f"Erro processando {video_key}: {e}")
        return None


def process_video_batch(args):
    """Processa um batch de vídeos (para paralelização)."""
    client, bucket, video_keys = args
    results = []
    for key in video_keys:
        meta = get_video_metadata(client, bucket, key)
        if meta:
            results.append(meta)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Gera Parquet de metadados para dataset de vídeos no S3"
    )
    parser.add_argument("--bucket", required=True, help="Nome do bucket S3")
    parser.add_argument("--prefix", default="", help="Prefixo/pasta no S3")
    parser.add_argument("--output", default="metadata.parquet", help="Arquivo de saída")
    parser.add_argument("--region", default="us-east-1", help="Região AWS")
    parser.add_argument("--endpoint-url", default=None, help="Endpoint URL (para R2, MinIO, etc)")
    parser.add_argument("--workers", type=int, default=16, help="Número de workers paralelos")
    parser.add_argument("--batch-size", type=int, default=100, help="Tamanho do batch por worker")
    parser.add_argument("--sample", type=int, default=None, help="Processar apenas N vídeos (para teste)")

    args = parser.parse_args()

    # Cria cliente S3
    client = get_s3_client(args.region, args.endpoint_url)

    # Lista todos os vídeos
    videos = list_s3_videos(client, args.bucket, args.prefix)

    if args.sample:
        import random
        videos = random.sample(videos, min(args.sample, len(videos)))
        print(f"Amostrando {len(videos)} vídeos para teste")

    # Divide em batches
    batches = []
    for i in range(0, len(videos), args.batch_size):
        batch = videos[i : i + args.batch_size]
        # Cada worker precisa de seu próprio cliente S3
        batches.append((get_s3_client(args.region, args.endpoint_url), args.bucket, batch))

    # Processa em paralelo
    all_metadata = []

    print(f"Processando {len(videos)} vídeos com {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_video_batch, batch) for batch in batches]

        with tqdm(total=len(videos), desc="Extraindo metadados") as pbar:
            for future in as_completed(futures):
                results = future.result()
                all_metadata.extend(results)
                pbar.update(len(results))

    # Cria DataFrame e salva
    df = pd.DataFrame(all_metadata)

    print(f"\nEstatísticas do dataset:")
    print(f"  Total de vídeos processados: {len(df)}")
    print(f"  Resoluções únicas: {df.groupby(['width', 'height']).size().shape[0]}")
    print(f"  Duração média: {df['duration_seconds'].mean():.2f}s")
    print(f"  Frames médios: {df['num_frames'].mean():.0f}")
    print(f"  FPS médio: {df['fps'].mean():.1f}")

    # Filtra vídeos muito curtos ou com problemas
    min_frames = 93  # Mínimo para LongCat
    df_valid = df[df["num_frames"] >= min_frames].copy()
    print(f"  Vídeos válidos (>= {min_frames} frames): {len(df_valid)}")

    # Salva Parquet
    df_valid.to_parquet(args.output, index=False)
    print(f"\nParquet salvo em: {args.output}")

    # Também salva uma versão CSV para inspeção manual
    csv_path = args.output.replace(".parquet", ".csv")
    df_valid.head(1000).to_csv(csv_path, index=False)
    print(f"Amostra CSV salva em: {csv_path}")


if __name__ == "__main__":
    main()

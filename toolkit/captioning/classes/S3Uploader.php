<?php

require 'vendor/autoload.php';

use Aws\S3\S3Client;
use Aws\Exception\AwsException;

class S3Uploader {
    /** @var S3Client */
    private $s3Client;
    /** @var string */
    private $bucket;
    /** @var string */
    private $vae_cache_prefix;
    /** @var string */
    private $image_data_prefix;
    /** @var string */
    private $text_cache_prefix;

    public function __construct($bucket, $region, $key, $secret, $endpoint, $vae_cache_prefix, $text_cache_prefix, $image_data_prefix) {
        $this->s3Client = new S3Client([
            'version' => 'latest',
            'region'  => $region,
            'credentials' => [
                'key'    => $key,
                'secret' => $secret,
            ],
            'endpoint' => $endpoint,
        ]);
        $this->bucket = $bucket;
        $this->vae_cache_prefix = $vae_cache_prefix;
        $this->text_cache_prefix = $text_cache_prefix;
        $this->image_data_prefix = $image_data_prefix;
    }

    /**
     * Upload a VAE cache file to S3, under the embed prefix as a .pt file.
     * 
     * A Client worker will POST this to us, we need to accept and forward to S3.
     */
    public function uploadVAECache($file, $key) {
        return $this->uploadFile($file, $this->vae_cache_prefix .'/'. $key);
    }

    /**
     * Upload an image file to S3, under the image data prefix as a .png file.
     * 
     * A Client worker will POST this to us, we need to accept and forward to S3.
     */
    public function uploadImage($file, $key) {
        return $this->uploadFile($file, $this->image_data_prefix .'/'. $key);
    }

    /**
     * Upload a text cache file to S3, under the embed prefix as a .pt file.
     * 
     * A Client worker will POST this to us, we need to accept and forward to S3.
     */
    public function uploadTextCache($file, $key) {
        return $this->uploadFile($file, $this->text_cache_prefix .'/'. $key);
    }

    public function uploadFile($file, $key) {
        try {
            $result = $this->s3Client->putObject([
                'Bucket' => $this->bucket,
                'Key'    => $key,
                'SourceFile'   => $file,
            ]);
            return $result['ObjectURL'];
        } catch (AwsException $e) {
            // Output error message if fails
            error_log($e->getMessage());
            return null;
        }
    }

    public function uploadContent($content, $key, $contentType = 'text/plain') {
        try {
            $result = $this->s3Client->putObject([
                'Bucket' => $this->bucket,
                'Key'    => $key,
                'Body'   => $content,
                'ContentType' => $contentType,
            ]);
            return $result['ObjectURL'];
        } catch (AwsException $e) {
            // Output error message if fails
            error_log($e->getMessage());
            return null;
        }
    }
}

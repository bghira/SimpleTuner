<?php

require_once __DIR__ . '/classes/Authorization.php';
require_once __DIR__ . '/classes/BackendController.php';
require_once __DIR__ . '/classes/S3Uploader.php';

$authorization = new Authorization('/var/www/.users.json');

// Load MySQL credentials
$mysql_credentials = json_decode(file_get_contents('/var/www/.mysql.json'), true);
$users = json_decode(file_get_contents('/var/www/.users.json'), true);

// Create PDO connection
$dsn = "mysql:host={$mysql_credentials['host']};dbname={$mysql_credentials['database']};charset=utf8mb4";

try {
	$pdo = new PDO($dsn, $mysql_credentials['user'], $mysql_credentials['password'], [
		PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
		PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
	]);
} catch (PDOException $e) {
	die('Could not connect to the database (' . $dsn . '): ' . $e->getMessage());
}

// Create the `dataset` table if it does not exist
$pdo->exec("CREATE TABLE IF NOT EXISTS dataset (
  data_id int AUTO_INCREMENT PRIMARY KEY,
  URL varchar(255) NOT NULL,
  pending tinyint(1) NOT NULL DEFAULT '0',
  result longtext,
  submitted_at datetime DEFAULT NULL,
  attempts int DEFAULT '0',
  error text,
  client_id varchar(255) DEFAULT NULL,
  updated_at datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  job_group varchar(255) DEFAULT NULL
)");

$aws_config = json_decode(file_get_contents('/var/www/.aws.json'), true);
$s3_uploader = new S3Uploader(
    $aws_config['aws_bucket_name'],
    $aws_config['aws_region'],
    $aws_config['aws_access_key_id'],
    $aws_config['aws_secret_access_key'],
    $aws_config['aws_endpoint_url'],
    $aws_config['vae_cache_prefix'],
    $aws_config['text_cache_prefix'],
    $aws_config['image_data_prefix']
);

$backendController = new BackendController($pdo, $s3_uploader);
$result = $backendController->handleRequest();

echo json_encode($result);
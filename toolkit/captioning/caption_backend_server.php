<?php

// Load MySQL credentials
$credentials = json_decode(file_get_contents("/var/www/.mysql.json"), true);
$users = json_decode(file_get_contents("/var/www/.users.json"), true);

// Create PDO connection
$dsn = "mysql:host={$credentials['host']};dbname={$credentials['database']};charset=utf8mb4";
try {
    $pdo = new PDO($dsn, $credentials['user'], $credentials['password'], [
        PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
        PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
    ]);
} catch (PDOException $e) {
    die("Could not connect to the database (". $dsn ."): " . $e->getMessage());
}

// Create the `dataset` table if it does not exist
$pdo->exec("CREATE TABLE IF NOT EXISTS dataset (
    data_id INT AUTO_INCREMENT PRIMARY KEY,
    URL VARCHAR(255) NOT NULL,
    pending BOOLEAN NOT NULL DEFAULT 0,
    result TEXT,
    submitted_at DATETIME,
    attempts INT DEFAULT 0,
    error TEXT
)");

// Action handling
$action = $_GET['action'] ?? '';
$client_id = $_GET['client_id'];
$secret = $_GET['secret'];

if (!in_array($client_id, array_keys($users)) || $secret !== $users[$client_id]) {
        http_response_code(403);
        echo "Unauthorized.";
        exit;
}

$filter_strings = [
    '</s>',
    'The image showcases '
];

try {
    switch ($action) {
        case 'list_jobs':
            $count = $_GET['count'] ?? 1;
            $total_jobs = $pdo->query("SELECT COUNT(*) FROM dataset")->fetchColumn();
            $remaining_jobs = $pdo->query("SELECT COUNT(*) FROM dataset WHERE pending = 0 AND result IS NULL")->fetchColumn();
            $completed_jobs = $total_jobs - $remaining_jobs;
            $stmt = $pdo->prepare("SELECT * FROM dataset WHERE pending = 0 OR (submitted_at IS NOT NULL AND submitted_at < NOW() - INTERVAL 1 HOUR) AND result IS NULL ORDER BY RAND() LIMIT ?");
            $stmt->bindValue(1, $count, PDO::PARAM_INT);
            $stmt->execute();
            $jobs = $stmt->fetchAll();
            
            // Update pending and submitted_at for retrieved jobs
           foreach ($jobs as $idx => $job) {
               $updateStmt = $pdo->prepare("UPDATE dataset SET pending = 1, submitted_at = NOW(), attempts = attempts + 1 WHERE data_id = ?");
               $updateStmt->execute([$job['data_id']]);
               $jobs[$idx]['total_jobs'] = $total_jobs;
               $jobs[$idx]['remaining_jobs'] = $remaining_jobs;
               $jobs[$idx]['completed_jobs'] = $completed_jobs;
           }
            
            echo json_encode($jobs);
            break;
        
        case 'submit_job':
            $dataId = $_REQUEST['job_id'] ?? '';
            $result = $_REQUEST['result'] ?? '';
            $caption_source = $_REQUEST['caption_source'] ?? '';
            $status = $_REQUEST['status'] ?? 'success';
            $error = $_REQUEST['error'] ?? '';

            if ($status == 'error' && !$error) {
                echo "Error message required for status 'error'";
                exit;
            }

            if (!$result || !$dataId) {
                echo "Job ID and result are required";
                exit;
            }

            // Clean any of the strings found in $filter_strings
            $result = str_replace($filter_strings, '', $result);

            $updateStmt = $pdo->prepare("UPDATE dataset SET client_id = ?, result = ?, pending = 0, error = ?, caption_source = ? WHERE data_id = ?");
            $updateStmt->execute([$client_id, $result, $error, $caption_source, $dataId]);
            echo "Job submitted successfully";
            break;

        default:
            echo "Invalid action";
    }
} catch (\Throwable $ex) {
    echo "An error occurred: " . $ex->getMessage();
}
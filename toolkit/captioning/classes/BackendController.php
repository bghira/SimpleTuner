<?php

/**
 * BackendController
 *
 * Accept the incoming REQUEST and parse it.
 */

class BackendController {
	/** @var PDO */
	private $pdo;
	/** @var string */
	private $action;
	/** @var string */
	private $error;
	/** @var string */
	private $client_id;
	/** @var string */
	private $job_type;
	/** @var S3Uploader */
	private $s3_uploader;

	public function __construct(PDO $pdo, S3Uploader $s3_uploader) {
		$this->pdo = $pdo;
		$this->s3_uploader = $s3_uploader;
		$this->getParameters();
	}

	public function getParameters() {
		// Action handling
		$this->action = $_REQUEST['action'] ?? '';
		$this->job_type = $_REQUEST['job_type'] ?? '';
		$this->error = $_REQUEST['error'] ?? '';
	}

	public function handleRequest() {
		return $this->{$this->action}();
	}

	public function list_jobs() {
		try {
			$limit = 500; // Number of rows to fetch and randomize in PHP
			$count = $_GET['count'] ?? 1; // Number of rows to actually return

			// Fetch the rows
			if ($this->job_type === 'vae') {
				$total_jobs = $this->pdo->query('SELECT COUNT(*) FROM dataset')->fetchColumn();
				$remaining_jobs = $this->pdo->query('SELECT COUNT(*) FROM dataset WHERE pending = 0')->fetchColumn();
				$stmt = $this->pdo->prepare('SELECT * FROM dataset WHERE pending = 0 LIMIT ?');
			} elseif ($this->job_type === 'dataset_upload') {
				$total_jobs = $this->pdo->query('SELECT COUNT(*) FROM dataset')->fetchColumn();
				$remaining_jobs = $this->pdo->query('SELECT COUNT(*) FROM dataset WHERE upload_pending = 0 AND result IS NULL')->fetchColumn();
				$stmt = $this->pdo->prepare('SELECT * FROM dataset WHERE result IS NULL LIMIT ?');
			}
			$stmt->bindValue(1, $limit, PDO::PARAM_INT);
			$stmt->execute();
			$jobs = $stmt->fetchAll();

			// Shuffle the array in PHP
			shuffle($jobs);

			// Slice the array to get only the number of rows specified by $count
			$jobs = array_slice($jobs, 0, $count);

			// Update the database for the selected jobs
			foreach ($jobs as $idx => $job) {
				if ($this->job_type === 'vae') {
					$updateStmt = $this->pdo->prepare('UPDATE dataset SET pending = 1, submitted_at = NOW(), attempts = attempts + 1 WHERE data_id = ?');
				} elseif ($this->job_type === 'dataset_upload') {
					$updateStmt = $this->pdo->prepare('UPDATE dataset SET upload_pending = 1 WHERE data_id = ?');
				}
				$updateStmt->execute([$job['data_id']]);
				$jobs[$idx]['total_jobs'] = $total_jobs;
				$jobs[$idx]['remaining_jobs'] = $remaining_jobs; // Update remaining jobs count
				$jobs[$idx]['completed_jobs'] = $total_jobs - $remaining_jobs;
				$jobs[$idx]['job_type'] = $this->job_type;
			}

			// Return the selected jobs
			return $jobs;
		} catch (\Throwable $ex) {
			echo 'An error occurred: ' . $ex->getMessage();
		}
	}

	public function submit_job() {
		try {
			$dataId = $_REQUEST['job_id'] ?? '';
			$result = $_REQUEST['result'] ?? '';
			$status = $_REQUEST['status'] ?? 'success';
			if (!$result || !$dataId) {
				echo 'Job ID and result are required';
				exit;
			}
			if ($status == 'error' && !$this->error) {
				echo "Error message required for status 'error'";
				exit;
			}

			if ($status !== 'error') {
				$stmt = $this->pdo->prepare('SELECT data_id FROM dataset WHERE data_id = ?');
				$stmt->execute([$dataId]);
				$filename = $stmt->fetchColumn();
				if (!$filename) {
					echo 'Job ID not found';
					exit;
				}
				if ($this->job_type === 'vae') {
					if (!in_array('result_file', array_keys($_FILES))) {
						echo 'Result files are required for VAE tasks.';
						echo 'Provided files: ' . json_encode($_FILES);
						exit;
					}
					$result = $this->s3_uploader->uploadVAECache($_FILES['result_file']['tmp_name'], $filename . '.pt');
					$updateStmt = $this->pdo->prepare('UPDATE dataset SET client_id = ?, error = ? WHERE data_id = ?');
					$updateStmt->execute([$this->client_id, $this->error, $dataId]);
				} elseif ($this->job_type === 'dataset_upload') {
					if (in_array('image_file', $_FILES)) $result = $this->s3_uploader->uploadImage($_FILES['image_file']['tmp_name'], $filename . '.png');
					$updateStmt = $this->pdo->prepare('UPDATE dataset SET result = ?, upload_pending = 1 WHERE data_id = ?');
					$updateStmt->execute([$result, $dataId]);
				} elseif ($this->job_type === 'text') {
					$result = $this->s3_uploader->uploadTextCache($_FILES['result_file']['tmp_name'], $filename);
				} else {
					echo 'Invalid job type: ' . $this->job_type . ' - must be "vae" or "text"';
					exit;
				}
			}
			return ['status' => 'success', 'result' => 'Job submitted successfully'];
		} catch (\Throwable $ex) {
			echo 'An error occurred for FILES ' . json_encode($_FILES) . ': ' . $ex->getMessage() . ', traceback: ' . $ex->getTraceAsString();
		}
	}
}

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

			$total_jobs = $this->pdo->query('SELECT COUNT(*) FROM dataset WHERE pending = 0 AND result IS NULL')->fetchColumn();

			// Fetch the rows
			$stmt = $this->pdo->prepare('SELECT * FROM dataset WHERE pending = 0 AND result IS NULL LIMIT ?');
			$stmt->bindValue(1, $limit, PDO::PARAM_INT);
			$stmt->execute();
			$jobs = $stmt->fetchAll();

			// Shuffle the array in PHP
			shuffle($jobs);

			// Slice the array to get only the number of rows specified by $count
			$jobs = array_slice($jobs, 0, $count);

			// Update the database for the selected jobs
			foreach ($jobs as $idx => $job) {
				$updateStmt = $this->pdo->prepare('UPDATE dataset SET pending = 1, submitted_at = NOW(), attempts = attempts + 1 WHERE data_id = ?');
				$updateStmt->execute([$job['data_id']]);
				$jobs[$idx]['total_jobs'] = $total_jobs;
				$jobs[$idx]['remaining_jobs'] = $total_jobs - count($jobs); // Update remaining jobs count
				$jobs[$idx]['completed_jobs'] = $total_jobs - $jobs[$idx]['remaining_jobs'];
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
				if (!in_array('image_file', array_keys($_FILES)) || !in_array('result_file', array_keys($_FILES))) {
					echo 'Image and result files are required.';
					if (in_array('result_file', $_FILES)) {
						echo ' Only the result file was provided.';
					}
					echo 'Provided files: ' . json_encode($_FILES);
					exit;
				}
				if ($this->job_type === 'vae') {
					$result = $this->s3_uploader->uploadVAECache($_FILES['result_file']['tmp_name'], $filename . '.pt');
					$result = $this->s3_uploader->uploadImage($_FILES['image_file']['tmp_name'], $filename . '.png');
				} elseif ($this->job_type === 'text') {
					$result = $this->s3_uploader->uploadTextCache($_FILES['result_file']['tmp_name'], $filename);
				} else {
					echo 'Invalid job type: ' . $this->job_type . ' - must be "vae" or "text"';
					exit;
				}
			}

			$updateStmt = $this->pdo->prepare('UPDATE dataset SET client_id = ?, result = ?, pending = 0, error = ? WHERE data_id = ?');
			$updateStmt->execute([$this->client_id, $result, $this->error, $dataId]);

			return ['status' => 'success', 'result' => 'Job submitted successfully'];
		} catch (\Throwable $ex) {
			echo 'An error occurred for FILES ' . json_encode($_FILES) . ': ' . $ex->getMessage() . ', traceback: ' . $ex->getTraceAsString();
		}
	}
}

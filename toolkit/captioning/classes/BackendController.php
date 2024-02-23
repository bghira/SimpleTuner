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
	private $client_id;
	/** @var string */
	private $job_type;

	public function __construct(PDO $pdo) {
		$this->pdo = $pdo;
		$this->getParameters();
		$this->handleRequest();
	}

	public function getParameters() {
		// Action handling
		$this->action = $_GET['action'] ?? '';
		$this->job_type = $_GET['job_type'] ?? '';
	}

	public function handleRequest() {
		return $this->{$this->action}();
	}

	public function list_jobs() {
		try {
			$count = $_GET['count'] ?? 1;
			$total_jobs = $this->pdo->query('SELECT COUNT(*) FROM dataset')->fetchColumn();
			$remaining_jobs = $this->pdo->query('SELECT COUNT(*) FROM dataset WHERE pending = 0 AND result IS NULL')->fetchColumn();
			$completed_jobs = $total_jobs - $remaining_jobs;
			$stmt = $this->pdo->prepare('SELECT * FROM dataset WHERE pending = 0 AND result IS NULL ORDER BY RAND() LIMIT ?');
			$stmt->bindValue(1, $count, PDO::PARAM_INT);
			$stmt->execute();
			$jobs = $stmt->fetchAll();

			foreach ($jobs as $idx => $job) {
				$updateStmt = $this->pdo->prepare('UPDATE dataset SET pending = 1, submitted_at = NOW(), attempts = attempts + 1 WHERE data_id = ?');
				$updateStmt->execute([$job['data_id']]);
				$jobs[$idx]['total_jobs'] = $total_jobs;
				$jobs[$idx]['remaining_jobs'] = $remaining_jobs;
				$jobs[$idx]['completed_jobs'] = $completed_jobs;
			}

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
			$error = $_REQUEST['error'] ?? '';

			if ($status == 'error' && !$error) {
				echo "Error message required for status 'error'";
				exit;
			}

			if (!$result || !$dataId) {
				echo 'Job ID and result are required';
				exit;
			}

			$updateStmt = $this->pdo->prepare('UPDATE dataset SET client_id = ?, result = ?, pending = 0, error = ? WHERE data_id = ?');
			$updateStmt->execute([$this->client_id, $result, $error, $dataId]);

			return ['status' => 'success', 'result' => 'Job submitted successfully'];
		} catch (\Throwable $ex) {
			echo 'An error occurred: ' . $ex->getMessage();
		}
	}
}

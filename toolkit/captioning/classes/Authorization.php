<?php

/**
 * Authorization - Handles the authorization of the client
 */

class Authorization {
	/** @var string */
	private $client_id;
	/** @var string */
	private $secret;
	/** @var string */
	private $user_config_path;
	/** @var array */
	private $users;

	public function __construct(string $user_config_path, bool $test_authorization = true) {
		// Retrieve client_id and secret from POST params:
		if (!isset($_REQUEST['client_id']) || !isset($_REQUEST['secret'])) {
			http_response_code(403);
			echo 'Unauthorized.';
			exit;
		}
		$this->client_id = $_REQUEST['client_id'];
		$this->secret = $_REQUEST['secret'];
		$this->user_config_path = $user_config_path;
		$this->load_user_database();
        if ($test_authorization) $this->authorize();
	}

    /**
     * Load the user database from disk.
     *
     * @return Authorization
     */
	private function load_user_database() {
		// Load the user database from the file:
		try {
			$this->users = json_decode(file_get_contents($this->user_config_path), true);
            return $this;
		} catch (Exception $e) {
            error_log($e->getMessage());
			http_response_code(500);
			echo 'Internal server error.';
			exit;
		}
	}

	public function authorize() {
		// Check if client_id and secret are valid:
		if (!in_array($this->client_id, array_keys($this->users)) || $this->secret !== $this->users[$this->client_id]) {
			http_response_code(403);
			echo 'Unauthorized.';
			exit;
		}
	}
}

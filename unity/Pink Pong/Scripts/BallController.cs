using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;

public class BallController : MonoBehaviour
{
    private Rigidbody ballRb;
    public TextMeshProUGUI playerScoreText;
    public TextMeshProUGUI enemyScoreText;
    public TextMeshProUGUI gameResultText;
    public TextMeshProUGUI countdownText;
    public GameObject easyButton;
    public GameObject hardButton;
    public GameObject playerBat;
    public GameObject enemyBat;
    public EnemyController enemyController;
    public float enemyMoveSpeed = 4.0f;
    public int playerScore = 0;
    public int enemyScore = 0;
    public int winningScore = 5;
    public bool playerAdvantage;
    public bool enemyAdvantage;
    public float ballSpeed = 3.0f; // Initialize speed of the ball
    public float ballSpeedAugment = 3.0f; // For sideways movements we augment the speed
    public float bounceSpeed = 4.0f; // The ball bounces after collision with upper and lower sides
    private List<float> randomY = new List<float>(new float[] {0.2f, -0.2f}); // Small random deviations of Y movement applied to middle sensor
    public Vector3 lowerSideCollisionForce = new Vector3(0.2f, 0.2f, 0); // Force applied when ball his the lower side
    public Vector3 upperSideCollisionForce = new Vector3(0.2f, -0.2f, 0); // Force applied when ball his the upper side
    public Vector3 playerUpperSensorForce = new Vector3(1, 0.8f, 0); // Force applied to the player's upper sensor
    public Vector3 playerLowerSensorForce = new Vector3(1, -0.8f, 0); // Force applied to the player's lower sensor
    public Vector3 enemyUpperSensorForce = new Vector3(-1, 0.8f, 0); // Force applied to the enemy's upper sensor
    public Vector3 enemyLowerSensorForce = new Vector3(-1, -0.8f, 0); // Force applied to the enemy's lower sensor
    //private Vector3[] randomForceDirection = {Vector3.left, Vector3.right};
    private Vector3[] randomForceDirection = {new Vector3(-2, 0, 0), new Vector3(2, 0, 0)};
    // Start is called before the first frame update
    public void StartGame(float difficulty)
    {
        Cursor.visible = false;
        gameObject.SetActive(true);
        transform.position = new Vector3(0, 500, 0); // We have to move the ball away since due to the coroutine we cannot simply deactivate it
        StartCoroutine(Countdown());
        
        enemyMoveSpeed *= difficulty;
    }

    // Update is called once per frame
    void Update()
    {   
        if (playerScore == winningScore || enemyScore == winningScore)
        {
            GameOver();
        }
    }
    void OnCollisionEnter(Collision other)
    {
        // Gameplay Mechanics: Forces and speeds applied after the diverse collisions

        if (other.gameObject.CompareTag("Upper Side"))
        {
            ballRb.AddForce(upperSideCollisionForce, ForceMode.Impulse);
        }

        if (other.gameObject.CompareTag("Lower Side"))
        {
            ballRb.AddForce(lowerSideCollisionForce, ForceMode.Impulse);
        }

        if (other.gameObject.CompareTag("Left Side"))
        {
            playerAdvantage = false;
            enemyAdvantage = true;
            UpdateScore();
            transform.position = new Vector3(0, 0, 0);
            ApplyRandomForceLeftOrRight();
        }

        if (other.gameObject.CompareTag("Right Side"))
        {
            playerAdvantage = true;
            enemyAdvantage = false;
            UpdateScore();
            transform.position = new Vector3(0, 0, 0);
            ApplyRandomForceLeftOrRight();
        }

        if (other.gameObject.CompareTag("Player Sensor Upper"))
        {
            ballRb.AddForce(playerUpperSensorForce * ballSpeed * ballSpeedAugment, ForceMode.Impulse);
        }

        if (other.gameObject.CompareTag("Player Sensor Middle"))
        {
            ballRb.AddForce(new Vector3(1, randomY[Random.Range(0, randomY.Count)], 0) * ballSpeed * ballSpeedAugment, ForceMode.Impulse);
        }

        if (other.gameObject.CompareTag("Player Sensor Lower"))
        {
            ballRb.AddForce(playerLowerSensorForce * ballSpeed * ballSpeedAugment, ForceMode.Impulse);
        }

        if (other.gameObject.CompareTag("Enemy Upper Sensor"))
        {
            ballRb.AddForce(enemyUpperSensorForce * ballSpeed * ballSpeedAugment, ForceMode.Impulse);
        }

        if (other.gameObject.CompareTag("Enemy Middle Sensor"))
        {
            ballRb.AddForce(new Vector3(-1, randomY[Random.Range(0, randomY.Count)], 0) * ballSpeed * ballSpeedAugment, ForceMode.Impulse);
        }

        if (other.gameObject.CompareTag("Enemy Lower Sensor"))
        {
            ballRb.AddForce(enemyLowerSensorForce * ballSpeed * ballSpeedAugment, ForceMode.Impulse);
        }
    }

    void UpdateScore()
    {
        if (playerAdvantage)
        {
            playerScore += 1;
            playerScoreText.text = "" + playerScore;
        }

        if (enemyAdvantage)
        {
            enemyScore += 1;
            enemyScoreText.text = "" + enemyScore;
        }
    }

    public void RestartGame()
    {
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    IEnumerator GameResultVisible()
    {
        // Display game result for a couple of secs, then reload scene
        yield return new WaitForSeconds(2);
        RestartGame();
    }

    IEnumerator Countdown()
    {
        easyButton.gameObject.SetActive(false);
        hardButton.gameObject.SetActive(false);

        // 'Animate' 3..2..1 countdown before the game actually starts 
        for (int i = 3; i > 0; i--)
        {
            countdownText.text = "" + i;
            countdownText.gameObject.SetActive(true);
            yield return new WaitForSeconds(1);
            countdownText.gameObject.SetActive(false);
            yield return new WaitForSeconds(1);
        }

        transform.position = new Vector3(0, 0, 0); // Transform the moved ball back into its default position
        playerBat.gameObject.SetActive(true);
        enemyBat.gameObject.SetActive(true);
        
        ballRb = GetComponent<Rigidbody>();
        ApplyRandomForceLeftOrRight();
        playerScoreText.text = "" + playerScore;
        enemyScoreText.text = "" + enemyScore;
    }

    void ApplyRandomForceLeftOrRight()
    {
        // At the start and if a round has been lost, the ball moves randomly towards either player or enemy
        ballRb.AddForce(randomForceDirection[Random.Range(0, randomForceDirection.Length)] * ballSpeed, ForceMode.Impulse); 
    }

    void GameOver()
    {
            transform.position = new Vector3(0, 500, 0); // We have to move the ball away since due to the coroutine we cannot simply deactivate it
            playerBat.gameObject.SetActive(false);
            enemyBat.gameObject.SetActive(false);
            gameResultText.gameObject.SetActive(true);
            if (playerAdvantage)
            {
                gameResultText.text = "You won!";
            }
            else if (enemyAdvantage)
            {
                gameResultText.text = "Enemy won!";
            }
            Cursor.visible = true;
            StartCoroutine(GameResultVisible());
    }
}

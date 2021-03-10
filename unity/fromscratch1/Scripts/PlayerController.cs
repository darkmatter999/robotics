using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class PlayerController : MonoBehaviour
{
    // Initialize the player's speed
    public float speed;
    // Initialize the player's (physical) forces of moving and jumping 
    public float jumpForce;
    public float moveForce;
    public bool gameOver = false;
    private bool isOnGround;
    // We start with a score of 0 since no enemies have yet been eliminated
    private int score = 0;
    // How long is the player allowed to play before the game is over (in seconds)?
    private float timer = 2000f;
    // Initialize a Rigidbody variable for the player
    private Rigidbody playerRb;
    // Optionally: Initialize a (prefab) Game Object containing a text with congratulations upon level passing
    // public GameObject levelSucceedText;

    // Initialize a GameObject (a 3D text with congratulations upon level passing)
    public GameObject congratsText;
    // Initialize a script containing instructions for displaying a text with congratulations upon level passing
    // public LevelSucceedDisplay levelSucceedDisplay;
    // Start is called before the first frame update
    void Start()
    {
        playerRb = GetComponent<Rigidbody>();
        // We get the component 'LevelSucceed Display' attached to Game Object 'congratsText
        // levelSucceedDisplay = congratsText.GetComponent<LevelSucceedDisplay>();
        congratsText.gameObject.SetActive(false);
    }

    // Update is called once per frame
    void Update()
    {
        // Let the timer run once the game has started
        timer -= Time.deltaTime;
        
        if (Input.GetKey(KeyCode.LeftArrow))
        {
            playerRb.AddForce(Vector3.right * moveForce);
        }
        if (Input.GetKey(KeyCode.RightArrow))
        {
            playerRb.AddForce(Vector3.left * moveForce);
        }
        if (Input.GetKey(KeyCode.UpArrow))
        {
            playerRb.AddForce(Vector3.back * moveForce);
        }
        if (Input.GetKey(KeyCode.DownArrow))
        {
            playerRb.AddForce(Vector3.forward * moveForce);
        }
        // In order to jump, the player has to be grounded, otherwise the Space Bar can be 'spammed' and the player jump endlessly.
        if (isOnGround)
        {
            if (Input.GetKeyDown(KeyCode.Space))
            {
                playerRb.AddForce(Vector3.up * jumpForce);
                isOnGround = false;
            }
        }
        // We call Game Over once the player has 'fallen over the boundary'. The player must stay within the confines of the game plane.
        if (transform.position.y < 0.2f)
        {
            gameOver = true;
            Debug.Log("Game Over!");
            Destroy(gameObject);
        }
        // Once having eliminated all enemies (hardcoded as 4), the level has been accomplished. We display a congrats text
        // There are three options for displaying this text:
        // 1. (as done here) Through a separate script handling the visibility of the 3D text. Here we just call the appropriate function within that script.
        // 2. By just instantiating a prefab (this is a quicker and more elegant option)
        // 3. By just setting the text as Active: False initially (in 'Start') and once level is accomplished, to Active: True, thus render.
        // This third option is the most elegant and concise way to display the level accomplishment 3D text.
        // If appropriate, we may load a new scene (level)
        if (score == 4)
        {
            Debug.Log("Well done! Level accomplished!");
            Destroy(gameObject);
            //Instantiate(levelSucceedText, transform.position, levelSucceedText.transform.rotation);
            // levelSucceedDisplay.RenderText();
            // SceneManager.LoadScene(sceneName: "Scene2");
            congratsText.gameObject.SetActive(true);
           
        }
        // Game Over if the player is out of time
        if (timer < 0f)
        {
            Debug.Log("Time out! Game Over!");
            Destroy(gameObject);
        }
        
    }
    // The enemies are 'triggers'. Once the player hits them, they are killed and the player is consequently awarded a point.
    void OnTriggerEnter(Collider other)
    {
        Destroy(other.gameObject);
        score += 1;
    }
    // If the player jumps and hits the ground again, he is back on ground
    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("Ground"))
        {
            isOnGround = true;
        }
    }
}

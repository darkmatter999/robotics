using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using TMPro;

public class CollisionController : MonoBehaviour
{
    public TextMeshProUGUI counterText;
    public TextMeshProUGUI timerText;
    public GameObject neutralBall;
    public GameObject colliderBall;
    public GameObject lowerSide;
    public GameObject successText;
    public GameObject timeoutText;
    public GameObject crate;
    private Rigidbody ballRb;
    public int crateFull = 20;
    public float borderZ = 13; // Balls are spawned randomly up to this Z position
    private int count;
    private int timer;
    private int gameRuntime = 60; // How long does one match run
    private float randomZ; // The balls are spawned at random Z positions

    // Start is called before the first frame update
    void Start()
    {
        StartCoroutine(Timer()); // We count down from gameRuntime to 0
        transform.position = new Vector3(lowerSide.transform.position.x, lowerSide.transform.position.y, lowerSide.transform.position.z);
        count = 0;
        counterText.text = "Count: " + count;
        colliderBall.gameObject.SetActive(true);
        ballRb = colliderBall.GetComponent<Rigidbody>();
        InvokeRepeating("SpawnBall", 0, 1);
    }

    // Update is called once per frame
    void Update()
    {
        // The collision sensor moves with the (lower side of the) crate.
        // The collision sensor is necessary because by design Unity does not allow child colliders to work independently
        // (i.e. only the parent compound collider can actually be referred to)
        transform.position = new Vector3(lowerSide.transform.position.x, lowerSide.transform.position.y, lowerSide.transform.position.z);
        // We call success when the crate is full and time is still left
        if (count == crateFull)
        {
            crate.gameObject.SetActive(false);
            colliderBall.gameObject.SetActive(false);
            successText.gameObject.SetActive(true);
            StartCoroutine(RestartGame());
        }
    }

    private void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.CompareTag("Ball"))
        {
            count += 1;
            counterText.text = "Count: " + count;
            Vector3 colliderBallPosition = other.gameObject.transform.position;
            Destroy(other.gameObject);
            // Instantiate an exact copy of the initial ball at the exact position of the initial ball. 
            // That way, double counting because of double collisions are avoided
            // This duplicate 'replaces' the original ball for aesthetic reasons. The player should see what is inside his glass crate
            Instantiate(neutralBall, colliderBallPosition, neutralBall.transform.rotation);
        }
    }

    private void SpawnBall()
    {
        randomZ = Random.Range(-borderZ, borderZ);
        Instantiate(colliderBall, new Vector3(colliderBall.transform.position.x, colliderBall.transform.position.y, randomZ), colliderBall.transform.rotation);
    }

    IEnumerator RestartGame()
    {
        // After showing either congrats or timeout text for 3 seconds, the scene is reloaded
        yield return new WaitForSeconds(3);
        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
    }

    IEnumerator Timer()
    {
        // For loop managing the timer. Once time is up, we show the timeout text and reload the scene
        for (timer = gameRuntime; timer > -1; timer--)
        {
            timerText.text = "Time: " + timer;
            yield return new WaitForSeconds(1);
        }
        timeoutText.gameObject.SetActive(true);
        crate.gameObject.SetActive(false);
        StartCoroutine(RestartGame());
    }
}

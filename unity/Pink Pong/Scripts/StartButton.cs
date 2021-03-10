using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class StartButton : MonoBehaviour
{
    private Button button;
    public GameObject ball;
    public BallController ballController;
    public float difficulty;
    // Start is called before the first frame update
    void Start()
    {
        button = GetComponent<Button>();
        button.onClick.AddListener(StartMatch);
        ballController = ball.gameObject.GetComponent<BallController>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    void StartMatch()
    {
        ballController.StartGame(difficulty);
    }
}

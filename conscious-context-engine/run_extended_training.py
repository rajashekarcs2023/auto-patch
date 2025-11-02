#!/usr/bin/env python3
"""
Extended Training Run with Complete Output Logging
Captures all terminal output to a timestamped file for documentation
"""
import asyncio
import sys
import os
from datetime import datetime
import subprocess
from dotenv import load_dotenv

load_dotenv()

def create_output_logger():
    """Create timestamped output file and logger"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_output_{timestamp}.log"
    
    print(f"🚀 Starting Extended Self-Improving Reasoning Training")
    print(f"📝 All output will be logged to: {log_filename}")
    print(f"⏰ Training started at: {datetime.now().isoformat()}")
    print("=" * 80)
    
    return log_filename

def run_training_with_logging():
    """Run the training and capture all output"""
    log_filename = create_output_logger()
    
    # Create header for the log file
    header = f"""
{'='*80}
SELF-IMPROVING REASONING CHAIN ENGINE - EXTENDED TRAINING LOG
{'='*80}
Training Session: {datetime.now().isoformat()}
Goal: Demonstrate complete reasoning evolution over 20 training steps
Log File: {log_filename}
{'='*80}

"""
    
    try:
        # Run the main training script and capture output
        print(f"🎯 Running extended training session...")
        print(f"📊 Target: 20 training steps with reasoning evolution")
        print(f"🔄 Expected duration: 30-45 minutes")
        print("=" * 80)
        
        # Use subprocess to run and capture both stdout and stderr
        with open(log_filename, 'w', encoding='utf-8') as log_file:
            # Write header
            log_file.write(header)
            log_file.flush()
            
            # Run the training process
            process = subprocess.Popen(
                [sys.executable, 'main_reasoning.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and file
            for line in process.stdout:
                # Print to console
                print(line, end='')
                # Write to log file
                log_file.write(line)
                log_file.flush()  # Ensure real-time writing
            
            # Wait for process to complete
            return_code = process.wait()
            
            # Write completion footer
            footer = f"""
{'='*80}
TRAINING SESSION COMPLETED
{'='*80}
End Time: {datetime.now().isoformat()}
Return Code: {return_code}
Status: {'SUCCESS' if return_code == 0 else 'ERROR'}
Log File: {log_filename}
{'='*80}
"""
            log_file.write(footer)
            print(footer)
            
            return log_filename, return_code
            
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n\nTRAINING INTERRUPTED BY USER AT: {datetime.now().isoformat()}\n")
        return log_filename, -1
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        with open(log_filename, 'a') as log_file:
            log_file.write(f"\n\nERROR OCCURRED: {e}\nTime: {datetime.now().isoformat()}\n")
        return log_filename, -2

def analyze_training_results(log_filename):
    """Analyze the training results from the log file"""
    print(f"\n📊 TRAINING ANALYSIS")
    print("=" * 50)
    
    try:
        with open(log_filename, 'r') as f:
            content = f.read()
            
        # Extract key metrics
        reasoning_evolutions = content.count("REASONING EVOLVED:")
        evolution_rate = content.count("Evolution Rate:")
        task_scores = content.count("Task Score:")
        benchmark_summaries = content.count("BENCHMARK SUMMARY")
        
        print(f"📈 Reasoning Evolutions Detected: {reasoning_evolutions}")
        print(f"🔄 Evolution Rate Calculations: {evolution_rate}")
        print(f"🎯 Task Evaluations: {task_scores}")
        print(f"📋 Benchmark Summaries: {benchmark_summaries}")
        
        # Look for performance improvements
        if "IMPROVEMENT:" in content:
            improvements = [line.strip() for line in content.split('\n') if 'IMPROVEMENT:' in line]
            print(f"📊 Performance Improvements Found: {len(improvements)}")
            for imp in improvements[-3:]:  # Show last 3 improvements
                print(f"   {imp}")
        
        # File size and content analysis
        file_size = os.path.getsize(log_filename)
        line_count = content.count('\n')
        
        print(f"\n📄 Log File Stats:")
        print(f"   Size: {file_size:,} bytes")
        print(f"   Lines: {line_count:,}")
        print(f"   Location: {os.path.abspath(log_filename)}")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")

def main():
    """Main execution function"""
    print("🧠 SELF-IMPROVING REASONING CHAIN ENGINE")
    print("Extended Training Session with Complete Logging")
    print("=" * 80)
    
    # Check if we're in the right directory
    if not os.path.exists('main_reasoning.py'):
        print("❌ Error: main_reasoning.py not found in current directory")
        print("Please run this script from the conscious-context-engine directory")
        return
    
    # Run the training
    log_filename, return_code = run_training_with_logging()
    
    # Analyze results
    if return_code == 0:
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        analyze_training_results(log_filename)
    else:
        print(f"\n⚠️  Training ended with code: {return_code}")
        print(f"📝 Check log file for details: {log_filename}")
    
    print(f"\n📋 Complete training log saved to: {os.path.abspath(log_filename)}")
    print("Ready for hackathon demo! 🚀")

if __name__ == "__main__":
    main()